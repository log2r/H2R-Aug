import os
import cv2
import math
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import sapien.core as sapien
from transforms3d.quaternions import mat2quat
import argparse
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

hand_joint_pose_dict = {
    "shoulder_pan_joint": 0.0, # yaw
    "shoulder_lift_joint": -1.05,
    "elbow_joint": 0.3, # pitch
    "wrist_1_joint": -1.75, # pitch（selected）
    "wrist_2_joint": -1.50,   # roll
    "wrist_3_joint": -0.05, # pitch
    "joint1": 0.0,
    "joint0": 0.0,  #fix
    "joint2": 0.0,
    "joint3": 0.0,
    "joint5": 0.0,
    "joint4": 0.0,  #fix
    "joint6": 0.0,
    "joint7": 0.0,
    "joint9": 0.0,
    "joint8": 0.0,  #fix
    "joint10": 0.0,
    "joint11": 0.0,
    "joint12": 0.0, # cal cross(vec,{0,1}) 
    "joint13": 0.0, #fix
    "joint14": 0.0,
    "joint15": 0.0,
}

gripper_joint_pose_dict = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.15,
    "elbow_joint": 0.6,
    "wrist_1_joint": -2.5,
    "wrist_2_joint": -1.40,
    "wrist_3_joint": -1.5,
    "finger_joint": 0.0,
    "left_inner_finger_joint": 0.0,
    "left_inner_knuckle_joint": 0.0,
    'right_outer_knuckle_joint':0.0,
    "right_inner_finger_joint": 0.0,
    "right_inner_knuckle_joint": 0.0
}


class CameraTools(object):
    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic

    @staticmethod
    def convert_wc_to_cc(joint_world, camera_intrinsic):
        joint_cam = []
        for pt in joint_world:
            pt = np.asarray(pt)
            rot = np.asarray(camera_intrinsic["R"])
            trans = np.asarray(camera_intrinsic["T"])
            R = rot  # ^W_{W_0} R
            R_ = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]
            ]) # ^{W_0}_C R
            R_ = R_.T
            R = R @ R_
            R_wc = np.eye(4)
            R_wc[:3, :3] = R
            R_wc[:3, 3] = trans.flatten()
            pt_cam_ = np.dot(np.linalg.inv(R_wc), np.append(pt,1))
            pt_cam = pt_cam_[:3]/pt_cam_[3]
            joint_cam.append(pt_cam)
        joint_cam = np.array(joint_cam)
        return joint_cam

    @staticmethod
    def __cam2pixel(cam_coord, f, c):
        u = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
        v = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
        d = cam_coord[..., 2]
        return u, v, d
 
    @staticmethod
    def convert_cc_to_ic(joint_cam, camera_intrinsic):
        root_idx = 0
        center_cam = joint_cam[root_idx]
        joint_num = len(joint_cam)
        f = camera_intrinsic["f"]
        c = camera_intrinsic["c"]
        joint_img = np.zeros((joint_num, 3))
        joint_img[:, 0], joint_img[:, 1], joint_img[:, 2] = CameraTools.__cam2pixel(joint_cam, f, c)
        joint_img[:, 2] = joint_img[:, 2] - center_cam[2]
        return joint_img


def cal_ang(vec1, vec2):
    mul = np.dot(vec1, vec2)
    l1 = np.linalg.norm(vec1)
    l2 = np.linalg.norm(vec2)

    cos = mul / (l1 * l2)
    cos = np.clip(cos, -1.0, 1.0)

    res = np.arccos(cos)
    return res

def get_link_pos(robot, link_name):
    for link in robot.get_links():
        if(link.get_name()) == link_name:
            return link.get_pose().p

def cal_campos(keypoints, robot, p=[0,0,0]):
    ham_x = keypoints[9] - keypoints[0]
    ham_y = keypoints[13] - keypoints[0]
    ham_x /= np.linalg.norm(ham_x)
    ham_y = np.cross(ham_x, ham_y)
    ham_y /= np.linalg.norm(ham_y)
    ham_z = np.cross(ham_x, ham_y)
    ham_z /= np.linalg.norm(ham_z)

    s_name = ['ee_link', 'mcp_joint_2', 'mcp_joint_3']
    s_pos = [[], [], []]
    for idx in range(3):
        s_pos[idx] = get_link_pos(robot, s_name[idx])  

    sap_x = s_pos[1] - s_pos[0]
    sap_y = s_pos[2] - s_pos[0]
    sap_x /= np.linalg.norm(sap_x)
    sap_y = np.cross(sap_x, sap_y)
    sap_y /= np.linalg.norm(sap_y)
    sap_z = np.cross(sap_x, sap_y)
    sap_z /= np.linalg.norm(sap_z)

    R = [ham_x, ham_y, ham_z]
    R = np.asarray(R).T
    t = keypoints[0]
    R_wh = np.eye(4)
    R_wh[:3, :3] = R
    R_wh[:3, 3] = t.flatten()
    R = [sap_x, sap_y, sap_z]
    R = np.asarray(R).T
    t = s_pos[0]
    R_ws = np.eye(4)
    R_ws[:3, :3] = R
    R_ws[:3, 3] = t.flatten()   

    R_wh = np.asarray(R_wh)
    R_ws = np.asarray(R_ws)
    R = R_ws @ np.linalg.inv(R_wh)
    res = np.dot(R, np.append(p,1))
    
    return res[:3] / res[3]


class Retargetor():
    def __init__(self,
            ur5_hand_urdf_path,
            ur5_gripper_urdf_path,
            device,
        ):
        self.engine = sapien.Engine()
        renderer = sapien.SapienRenderer(device=device)
        self.engine.set_renderer(renderer)
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        self.scene.set_timestep(1 / 240.0)
        self.scene.add_ground(0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        # self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        self.loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        self.robot_hand: sapien.Articulation = self.loader.load(ur5_hand_urdf_path)
        self.robot_gripper: sapien.Articulation = self.loader.load(ur5_gripper_urdf_path)
    def __call__(self, 
            inpainted_img, 
            sim_cfg,
            eef_type):
        
        for idx, target in enumerate(sim_cfg):
            if target['is_right'] == 0:
                continue

            keypoints_3d = np.array(target['pred_keypoints_3d'])[0]
            cam_t = np.array(target['pred_cam_t'])[0]
            for pt in keypoints_3d:
                pt += cam_t

            for i in range(16):
                if i % 4 ==0:
                    continue
                rt = i + 4
                if i >= 13:
                    rt = i - 12
                l = rt - 1
                r = rt + 1
                if i % 4 == 1:
                    l = 0
                vec1 = keypoints_3d[r] - keypoints_3d[rt]
                vec2 = keypoints_3d[rt] - keypoints_3d[l]
                hand_joint_pose_dict[f"joint{i}"] = cal_ang(vec1, vec2)

            hand_joint_pose_dict['joint12'] = hand_joint_pose_dict['joint13']
            hand_joint_pose_dict['joint13'] = 0

            self.robot_hand.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
            self.robot_hand.set_qpos(list(hand_joint_pose_dict.values()))
            self.robot_gripper.set_root_pose(sapien.Pose([0, 0, 5], [1, 0, 0, 0]))
            self.robot_gripper.set_qpos(list(gripper_joint_pose_dict.values()))

            near, far = 0.1, 1e12
            width = int(target['width'])
            height = int(target['height'])
            fovy = 2 * np.arctan(height / (2 * target['focal_length']))
            camera = self.scene.add_camera(
                name="camera",
                width=width,
                height=height,
                fovy=fovy,
                near=near,
                far=far,
            )
            cam_pos = cal_campos(keypoints_3d, self.robot_hand)
            print('cam_pos', cam_pos)
            cam_pos *= 1.5
            tar_pos = get_link_pos(self.robot_hand, 'ee_link') # pos

            dir = tar_pos - cam_pos
            forward = dir / np.linalg.norm(dir)
            left = np.cross([0, 0, 1], forward)
            left = left / np.linalg.norm(left)
            up = np.cross(forward, left)
            up = up / np.linalg.norm(up)
            mat33 = np.stack([forward, left, up], axis=1)
            camera.set_pose(
                sapien.Pose(
                    p=cam_pos,
                    # p=cam_pos - dir * 0.5,
                    q=mat2quat(mat33)
                )
            )
            light = self.scene.add_directional_light(dir.tolist(), [2.8, 2.8, 2.8])
            
            cam_I = camera.get_intrinsic_matrix()
            f_x = cam_I[0, 0]
            f_y = cam_I[1, 1]
            c_x = cam_I[0, 2]
            c_y = cam_I[1, 2]
            camera_intrinsic = {
                "R": mat33,
                "T": cam_pos,
                "f": [f_x, f_y],
                "c": [c_x, c_y],
            }
            cam_tools = CameraTools(camera_intrinsic)

            wrist = get_link_pos(self.robot_hand, 'ee_link')
            thumb_tip = get_link_pos(self.robot_hand, 'index_tip_head')
            o_points = [wrist, thumb_tip]

            o_points = np.array(o_points)
            o_points = cam_tools.convert_wc_to_cc(o_points, camera_intrinsic)
            o_points = cam_tools.convert_cc_to_ic(o_points, camera_intrinsic)
            o_points = o_points[:, :2].astype(int).tolist()
            
            if eef_type:
                self.robot_hand.set_root_pose(sapien.Pose([0, 0, 7], [1, 0, 0, 0]))
                self.robot_gripper.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
            
            self.scene.step() 
            self.scene.update_render()
            camera.take_picture()
            rgba = camera.get_float_texture('Color') 
            seg_labels = camera.get_uint32_texture('Segmentation')
            robot_mask = (seg_labels[..., 0] > 1).astype(np.uint8) * 255
            bgr_img = (rgba * 255).clip(0, 255).astype("uint8")[..., [2, 1, 0]]

            Image.fromarray((rgba * 255).clip(0, 255).astype("uint8")).save(f"img.png")
            Image.fromarray((rgba * robot_mask[..., None]).clip(0, 255).astype("uint8")).save(f"label00.png")

            # calculate and place mask
            keypoints = np.array(target['keypoint'])
            wrist_pixel = keypoints[0]
            thumb_pixel = keypoints[8]
            
            hand_length = math.sqrt((wrist_pixel[0] - thumb_pixel[0]) ** 2 + (wrist_pixel[1] - thumb_pixel[1]) ** 2)
            arm_start = o_points[0]
            arm_end = o_points[1]
            arm_length = math.sqrt((arm_start[0] - arm_end[0]) ** 2 + (arm_start[1] - arm_end[1]) ** 2)
            scale_factor = hand_length / arm_length
            scale_factor = np.clip(scale_factor, 0.01, 9.99)
            robot_mask_h, robot_mask_w = robot_mask.shape[:2]
            new_size = (int(robot_mask_w * scale_factor), int(robot_mask_h * scale_factor))
            
            mask_rgb_scaled = cv2.resize(bgr_img, new_size, interpolation=cv2.INTER_LINEAR)
            mask_scaled = cv2.resize(robot_mask, new_size, interpolation=cv2.INTER_LINEAR)

            arm_end_scaled = (int(arm_end[0] * scale_factor), int(arm_end[1] * scale_factor))
            offset_x = thumb_pixel[0] - arm_end_scaled[0]
            offset_y = thumb_pixel[1] - arm_end_scaled[1]
            
            mask_h, mask_w = mask_scaled.shape[:2]
            target_h, target_w = inpainted_img.shape[:2]
            x_end = min(offset_x + mask_w, target_w)
            y_end = min(offset_y + mask_h, target_h)
            x_start = max(0, offset_x)
            y_start = max(0, offset_y)
            x_start_in_mask = max(0, -offset_x)
            y_start_in_mask = max(0, -offset_y)
            x_end_in_mask = x_start_in_mask + (x_end - x_start)
            y_end_in_mask = y_start_in_mask + (y_end - y_start)
            for c in range(0, 3):
                inpainted_img[y_start:y_end, x_start:x_end, c] = np.where(
                    mask_scaled[y_start_in_mask:y_end_in_mask, x_start_in_mask:x_end_in_mask] ==255,
                    mask_rgb_scaled[y_start_in_mask:y_end_in_mask, x_start_in_mask:x_end_in_mask, c],
                    inpainted_img[y_start:y_end, x_start:x_end, c]
                )
            self.scene.remove_camera(camera)
            self.scene.remove_light(light)
            return inpainted_img

def parse_args():
    parser = argparse.ArgumentParser(
        description="H2R single picture retarget script")
    parser.add_argument(
        "--image",
        default="/home/log2r/hand_teleop_real_sim_mix_adr/corl_sample/cfg/7/image.png"
    )
    parser.add_argument(
        "--cfg",
        default="/home/log2r/hand_teleop_real_sim_mix_adr/corl_sample/cfg/7/sim_cfg.json"
    )
    parser.add_argument(
        "--output",
        default="/home/log2r/hand_teleop_real_sim_mix_adr/corl_sample/res/7_.png"
    )
    parser.add_argument(
        "--eef_type",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    retargetor = Retargetor(
        ur5_hand_urdf_path="/home/log2r/hand_teleop_real_sim_mix_adr/ur5_leaphand/ur5_leaphand.urdf",
        ur5_gripper_urdf_path="/home/log2r/hand_teleop_real_sim_mix_adr/rebuttal/ur5/ur5_leaphand/ur5_gripper.urdf",
        device=args.device,
    )

    with open(args.cfg, "r") as f:
        sim_cfg = json.load(f)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {args.image}")

    result = retargetor(img, sim_cfg, args.eef_type)
    cv2.imwrite(args.output, result)
    print(f"Saved to {args.output}")
