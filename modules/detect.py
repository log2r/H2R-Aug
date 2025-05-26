import cv2
import torch
import numpy as np
from vitpose_model import ViTPoseModel

# from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig

class Detector():
    def __init__(self,
        hamer_cache_dir,
        hamer_checkpoint_path,
        detectron2_cfg_path,
        detectron2_checkpoint_path,
        device,
    ):
        download_models(hamer_cache_dir)
        self.model, self.model_cfg = load_hamer(hamer_checkpoint_path)
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
        detectron2_cfg = LazyConfig.load(detectron2_cfg_path)
        detectron2_cfg.train.init_checkpoint = detectron2_checkpoint_path
        self.detector = DefaultPredictor_Lazy(detectron2_cfg, self.device)
        self.cpm = ViTPoseModel(self.device)
    
    def __call__(self, img_path: str):
        img_cv2 = cv2.imread(img_path)

        # Detect humans in image
        det_out = self.detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        
        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(
            img, 
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        )

        if len(vitposes_out) == 0:
            # print("No keypoints detected.")
            return {}
        
        bboxes = []
        is_right = []
        keypoints_pixel = [] #

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.45
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
                keypoints_pixel.append(right_hand_keyp[:, :2].astype(int))
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.45
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)
                keypoints_pixel.append(right_hand_keyp[:, :2].astype(int))
        
        if not bboxes:
            return {}
        
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        keypoints_pixel = np.stack(keypoints_pixel)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        result_data = []
        for i, batch in enumerate(dataloader):
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            img_size = batch['img_size'].float().squeeze(0)
            target_data = {
                'bbox': boxes[i].tolist(),
                'keypoint': keypoints_pixel[i].tolist(),
                'is_right': int(right[i]),
                'pred_keypoints_3d': out['pred_keypoints_3d'].detach().cpu().numpy().tolist(),
                'pred_cam_t': out['pred_cam_t'].detach().cpu().numpy().tolist(),
                'focal_length': self.model_cfg.EXTRA.FOCAL_LENGTH,
                'width': img_size[0].item(),
                'height': img_size[1].item()
            }
            result_data.append(target_data)
        
        return result_data


