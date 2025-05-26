import os
import cv2
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
import csv
import hashlib
import traceback
import draccus

import hamer
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import DEFAULT_CHECKPOINT
from modules.detect import Detector
from modules.remove import Remover
from modules.retarget import Retargetor

@dataclass
class ImageCutConfig:
    hamer_cache_dir: str = CACHE_DIR_HAMER
    hamer_checkpoint: str = DEFAULT_CHECKPOINT
    detectron2_cfg: str = str(Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py')
    detectron2_checkpoint: str = "third_party/hamer/_DATA/model_final_f05665.pkl"
    ur5_hand_urdf_path: str = "ur5_leaphand/ur5_leaphand.urdf"
    ur5_gripper_urdf_path: str = "ur5_leaphand/ur5_gripper.urdf"
    sam_ckpt: str = "third_party/Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth"
    lama_config: str = "third_party/Inpaint_Anything/lama/configs/prediction/default.yaml"
    lama_ckpt: str = "third_party/Inpaint_Anything/pretrained_models/big-lama"

    output_path: str = "run/output/"
    log_dir: str = "run/log/"

    # device: str = "cuda:1"
    gpu_num: int = 1
    rank: int = 0
    threading_num: int = 1

class H2R():
    def __init__(self,
        cfg: ImageCutConfig
    ):
        self.device = f"cuda:{cfg.rank % cfg.gpu_num}"
        self.cfg = cfg
        self.detector = Detector(
            hamer_cache_dir = cfg.hamer_cache_dir,
            hamer_checkpoint_path = cfg.hamer_checkpoint,
            detectron2_cfg_path = cfg.detectron2_cfg,
            detectron2_checkpoint_path = cfg.detectron2_checkpoint,
            device = self.device
        )
        self.remover = Remover(
            sam_ckpt = cfg.sam_ckpt,
            lama_config = cfg.lama_config,
            lama_ckpt = cfg.lama_ckpt,
            device = self.device
        )
        self.retargetor = Retargetor(
            ur5_hand_urdf_path = cfg.ur5_hand_urdf_path,
            ur5_gripper_urdf_path = cfg.ur5_gripper_urdf_path,
            device = self.device,
        )
        os.makedirs(self.cfg.output_path, exist_ok=True)
        os.makedirs(self.cfg.log_dir, exist_ok=True)

    def __call__(self, img_path):
        output_path = os.path.join(self.cfg.output_path, os.path.basename(img_path))
        sim_cfg = self.detector(img_path)
        if sim_cfg == {}:
            return False
        inpainted_img = self.remover(img_path, sim_cfg)
        inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_RGBA2BGR)
        img = self.retargetor(inpainted_img, sim_cfg, 0) # 0 leaphand， 1 gripper
        cv2.imwrite(output_path, img)
        return True

if __name__ == "__main__":
    cfg = ImageCutConfig
    h2r = H2R(cfg)
    h2r('examples/images/19.jpg')
