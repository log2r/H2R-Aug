import os
import sys
import yaml

import torch
import cv2
import numpy as np
from omegaconf import OmegaConf

sys.path.append("third_party/Inpaint_Anything")
from segment_anything import SamPredictor, sam_model_registry
sys.path.append("third_party/Inpaint_Anything/lama")
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo


class Remover():
    def __init__(self,
        sam_ckpt,
        lama_config,
        lama_ckpt,
        dilate_factor: int = 5,
        mask_level: int = 1,
        device: str = 'cuda:0',
    ):
        if mask_level not in [0, 1, 2]:
            raise ValueError("mask_level must be 0, 1, or 2")
        
        self.dilate_factor = dilate_factor
        self.mask_level = mask_level
        self.device = device
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        predict_config = OmegaConf.load(lama_config)
        predict_config.model.path = lama_ckpt
        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        checkpoint_path = os.path.join(
            predict_config.model.path, 'models',
            predict_config.model.checkpoint
        )
        self.lama_train_config = train_config
        self.lama_predict_config = predict_config
        self.lama_model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        self.lama_model.freeze()
        if not predict_config.get('refine', False):
            self.lama_model.to(self.device)

    def __call__(self, img_path, sim_cfg):
        img = cv2.imread(str(img_path))
        left_flag, right_flag = False, False
        for i in range(len(sim_cfg)):
            if right_flag and left_flag:
                break
            if sim_cfg[i]["is_right"]:
                right_flag = True
            else:
                left_flag = True
            point_coords = np.array([sim_cfg[i]['keypoint'][1]])

            point_labels = np.array([1])
            self.predictor.set_image(image = img, image_format = "BGR")
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            masks = masks.astype(np.uint8) * 255
            mask = cv2.dilate(
                masks[self.mask_level].astype(np.uint8),
                np.ones((self.dilate_factor, self.dilate_factor), np.uint8),
                iterations=1
            )

            img = torch.from_numpy(img).float().div(255.)
            mask = torch.from_numpy(mask).float()
            batch = {
                'image': pad_tensor_to_modulo(img = img.permute(2, 0, 1).unsqueeze(0), mod = 8),
                'mask': (pad_tensor_to_modulo(img = mask.unsqueeze(0).unsqueeze(0), mod = 8) > 0) * 1,
            }
            batch = move_to_device(batch, self.device)
            batch = self.lama_model(batch)
            cur_res = batch[self.lama_predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

            h, w, _ = img.shape
            img = np.clip(cur_res[:h, :w] * 255, 0, 255).astype('uint8')
        
        return img

if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import random
    remover = Remover(
        sam_ckpt="third_party/Inpaint_Anything/pretrained_models/sam_vit_h_4b8939.pth",
        lama_config="third_party/Inpaint_Anything/lama/configs/prediction/default.yaml",
        lama_ckpt="third_party/Inpaint_Anything/pretrained_models/big-lama",
        device='cuda:0',
    )
    img_path = ""
    sim_config_path = ""
    output_path = ""
    with open(sim_config_path, 'r') as f:
        sim_cfg = json.load(f)
    img = remover(img_path, sim_cfg)
    cv2.imwrite(output_path, img)

    