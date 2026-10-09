"""Typed configuration. The defaults live here; override any field on the command line, e.g.

python -m gloved_hands train arm=b seed=1 train.lr=3e-4
"""

from dataclasses import dataclass, field
from typing import cast

from omegaconf import OmegaConf


@dataclass
class PathsConfig:
    data: str = "${oc.env:GLOVED_HANDS_DATA,/data/gloved-hands}"
    recordings: str = "${paths.data}/recordings"  # root passed to Recording(root, session)
    labels: str = "${paths.data}/labels"
    shards: str = "${paths.data}/shards"
    keyframes: str = "${paths.data}/keyframes"  # images/ to annotate, annotations.json from CVAT
    gloves: str = "${paths.data}/gloves"  # <variant>/<name>.png + <name>_mask.png: the glove alone, in the rig
    checkpoints: str = "${paths.data}/checkpoints"
    results: str = "${paths.data}/results"  # CSV and JSON written by validate and evaluate
    mano: str = "${paths.data}/models/mano"  # MANO_RIGHT.pkl, MANO_LEFT.pkl
    mano_mean_params: str = "${paths.data}/models/mano/mano_mean_params.npz"  # ships with WiLoR and HaMeR
    wilor: str = "${paths.data}/models/wilor/wilor_final.ckpt"
    wilor_config: str = "${paths.data}/models/wilor/model_config.yaml"
    detector: str = "${paths.data}/models/wilor/detector.pt"
    hamer: str = "${paths.data}/models/hamer/hamer_ckpts/checkpoints/hamer.ckpt"
    hamer_config: str = "${paths.data}/models/hamer/hamer_ckpts/model_config.yaml"
    sam2: str = "facebook/sam2-hiera-large"


@dataclass
class FitConfig:
    """Multi-view MANO fitting (labelling)."""

    sides: list[str] = field(default_factory=lambda: ["right", "left"])
    frame_stride: int = 3  # label every n-th frame
    chunk_size: int = 32  # frames optimised together
    mask_points: int = 300  # glove-mask pixels per view for the silhouette term
    depth_points: int = 300  # depth points per view for the depth term
    erosion_px: int = 7  # depth only from the eroded mask interior
    sigma_px: float = 10.0  # robust scale of the image terms; the reprojection term has weight 1
    sigma_depth_m: float = 0.01
    w_silhouette: float = 0.5
    w_depth: float = 1.0
    w_smooth_rotation: float = 1.0
    w_smooth_translation: float = 100.0
    w_pose_prior: float = 0.001
    w_shape_prior: float = 0.01
    iters_placement: int = 100  # orientation and translation only, reprojection only
    iters_full: int = 300
    lr: float = 0.02
    lr_translation: float = 0.002
    max_reprojection_px: float = 12.0  # QC: frames above either limit never reach training
    max_depth_mm: float = 15.0
    lovo_chunks: int = 5  # chunks refitted per camera for leave-one-view-out
    device: str = "cuda"


@dataclass
class DataConfig:
    sessions: list[str] = field(default_factory=list)  # the cohort's recordings (Arm A)
    held_out_task: str | None = None  # never trained on; tested separately
    n_images: int = 20000  # training images of every fold and of Arm B (matched budget)
    val_fraction: float = 0.1  # held out for early stopping: share of training recordings (A), of n_images (B)
    keyframes_per_session: int = 40
    glove_variants: list[str] = field(default_factory=lambda: ["nitrile-blue"])


@dataclass
class TrainConfig:
    """Identical for both arms."""

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_steps: int = 20000
    val_every: int = 1000
    patience: int = 5  # validations without improvement of val/pa_mpjpe
    rotation_deg: float = 30.0
    scale_jitter: float = 0.15
    workers: int = 6
    precision: str = "bf16-mixed"


@dataclass
class WandbConfig:
    """Training curves only, to follow runs from anywhere; everything else stays local."""

    project: str = "gloved-hands"
    mode: str = "online"  # online | offline | disabled


@dataclass
class Config:
    arm: str = "a"  # a: self-labelled rig recordings, b: recoloured DexYCB
    fold: int = 0  # leave-one-subject-out fold: test subject = sorted(subjects)[fold]
    seed: int = 0
    session: str | None = None  # recording for label, keyframes and validate
    model: str = "wilor"  # evaluate without adapter: zero-shot wilor or hamer
    adapter: str | None = None  # evaluate: LoRA checkpoint written by train
    paths: PathsConfig = field(default_factory=PathsConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def load_config(overrides: list[str]) -> Config:
    """Defaults merged with ``key=value`` overrides; values are type-checked."""
    merged = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.from_dotlist(overrides))
    return cast(Config, OmegaConf.to_object(merged))
