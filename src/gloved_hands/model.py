"""The released WiLoR and HaMeR models, and WiLoR fine-tuned with LoRA (identical for both arms)."""

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from peft import LoraConfig, inject_adapter_in_model
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torch import Tensor

from gloved_hands.config import PathsConfig, TrainConfig
from gloved_hands.geometry import rotvec_to_matrix
from gloved_hands.metrics import hand_errors

LORA_TARGETS = r"blocks\.\d+\.attn\.(qkv|proj)"  # attention of the ViT backbone; the refinement head stays frozen
LOSS_WEIGHTS = {  # those of HaMeR and WiLoR
    "keypoints_2d": 0.01,
    "keypoints_3d": 0.05,
    "global_orient": 0.001,
    "hand_pose": 0.001,
    "betas": 0.0005,
}


def load_pretrained(name: str, paths: PathsConfig) -> Any:
    """WiLoR or HaMeR with the released weights. Unlike the upstream loaders, this sets the MANO paths and skips
    the renderer, which needs OpenGL. Both models share ``forward_step({"img": crops})`` and its outputs."""
    if name == "wilor":
        from wilor.configs import get_config as get_wilor_config
        from wilor.models import WiLoR

        model, cfg, checkpoint = WiLoR, get_wilor_config(paths.wilor_config), paths.wilor
    elif name == "hamer":
        from hamer.configs import get_config as get_hamer_config
        from hamer.models import HAMER

        model, cfg, checkpoint = HAMER, get_hamer_config(paths.hamer_config), paths.hamer
    else:
        raise ValueError(f"unknown model {name!r}")
    cfg.defrost()
    cfg.MODEL.BBOX_SHAPE = [192, 256]  # crop width seen by the ViT, as in the upstream loaders
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    cfg.MANO.MODEL_PATH, cfg.MANO.MEAN_PARAMS = paths.mano, paths.mano_mean_params
    cfg.freeze()
    return model.load_from_checkpoint(checkpoint, strict=False, cfg=cfg, init_renderer=False)


class HandPoseModel(pl.LightningModule):
    """WiLoR with LoRA adapters on its backbone's attention. Only the adapters are trained and saved."""

    def __init__(self, cfg: TrainConfig, paths: PathsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = load_pretrained("wilor", paths)
        lora = LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=LORA_TARGETS
        )
        inject_adapter_in_model(lora, self.net.backbone)
        for name, parameter in self.net.named_parameters():
            parameter.requires_grad = "lora_" in name
        self.net.refine_net.eval()  # keeps its BatchNorm statistics as released: they are not saved with the adapters

    @classmethod
    def from_adapter(cls, path: str, cfg: TrainConfig, paths: PathsConfig) -> "HandPoseModel":
        model = cls(cfg, paths)
        state = torch.load(path, map_location="cpu")["state_dict"]
        if not state or model.load_state_dict(state, strict=False).unexpected_keys:
            raise ValueError(f"{path} does not hold adapters for this model")
        return model

    def forward(self, images: Tensor) -> dict[str, Any]:
        return self.net.forward_step({"img": images})

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        losses = self.losses(self(batch["image"]), batch)
        self.log_dict({f"train/{name}": value for name, value in losses.items()}, batch_size=len(batch["image"]))
        return losses["total"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        predicted = self(batch["image"])["pred_keypoints_3d"].float().cpu().numpy()
        errors = [hand_errors(p, t)["pa_mpjpe"] for p, t in zip(predicted, batch["joints"].cpu().numpy())]
        self.log("val/pa_mpjpe", float(np.mean(errors)), batch_size=len(errors))

    def losses(self, out: dict[str, Any], batch: dict[str, Any]) -> dict[str, Tensor]:
        """HaMeR's training losses without the adversarial term, averaged over the batch."""
        n, mano = len(batch["image"]), out["pred_mano_params"]
        predicted_3d = out["pred_keypoints_3d"] - out["pred_keypoints_3d"][:, :1]  # root-relative
        target_3d = batch["joints"] - batch["joints"][:, :1]
        target_orient = rotvec_to_matrix(batch["global_orient"])  # the model predicts rotation matrices
        target_pose = rotvec_to_matrix(batch["hand_pose"].reshape(n, 15, 3))
        terms = {
            "keypoints_2d": (out["pred_keypoints_2d"] - batch["keypoints"]).abs().sum((1, 2)).mean(),
            "keypoints_3d": (predicted_3d - target_3d).abs().sum((1, 2)).mean(),
            "global_orient": (mano["global_orient"].reshape(n, 3, 3) - target_orient).square().sum((1, 2)).mean(),
            "hand_pose": (mano["hand_pose"].reshape(n, 15, 3, 3) - target_pose).square().sum((1, 2, 3)).mean(),
            "betas": (mano["betas"].reshape(n, -1) - batch["betas"]).square().sum(1).mean(),
        }
        terms["total"] = torch.stack([LOSS_WEIGHTS[name] * value for name, value in terms.items()]).sum()
        return terms

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        adapters = [p for p in self.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(adapters, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, self.cfg.max_steps)
        return {"optimizer": optimiser, "lr_scheduler": {"scheduler": schedule, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Keep only the adapters: megabytes instead of gigabytes, and WiLoR's weights (CC-BY-NC-ND) stay out."""
        checkpoint["state_dict"] = {name: value for name, value in checkpoint["state_dict"].items() if "lora_" in name}
