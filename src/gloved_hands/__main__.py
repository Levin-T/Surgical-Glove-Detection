"""python -m gloved_hands <stage> [key=value ...]

label       session=<id>                   MANO labels of both hands of a recording
keyframes   session=<id>                   images of labelled frames to annotate in CVAT
validate    session=<id>                   validation layers of the labels (RQ2)
build-a     fold=<k>                       Arm A shards of one fold, and the fold's test sets
build-b                                    Arm B shards (recoloured DexYCB)
train       arm=<a|b> fold=<k> seed=<s>    WiLoR + LoRA on one arm (Arm B ignores the fold)
evaluate    fold=<k> adapter=<checkpoint>  a trained model on the fold's test sets (RQ1)
evaluate    fold=<k> model=<wilor|hamer>   a zero-shot baseline on the same test sets
"""

import csv
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from gloved_hands.capture import Recording
from gloved_hands.config import Config, load_config
from gloved_hands.datasets import ArmA, ArmB
from gloved_hands.evaluation import Evaluator
from gloved_hands.keyframes import Keyframes
from gloved_hands.labelling import Labeller, Labels
from gloved_hands.metrics import summarise
from gloved_hands.model import HandPoseModel
from gloved_hands.samples import random_example, read_shards, to_example
from gloved_hands.validation import LabelValidator


def label(cfg: Config) -> None:
    recording, labeller = _recording(cfg), Labeller(cfg)
    for side in cfg.fit.sides:
        labels = labeller.label(recording, side)
        labels.save(_labels_path(cfg, recording.session, side))
        print(
            f"{recording.session} {side}: {labels.passed.mean():.0%} passed QC, "
            f"median reprojection {np.median(labels.reprojection_px):.1f} px, "
            f"median depth residual {np.nanmedian(labels.depth_mm):.1f} mm"
        )


def keyframes(cfg: Config) -> None:
    recording = _recording(cfg)
    rng = np.random.default_rng(cfg.seed)
    frames: set[int] = set()
    for side in cfg.fit.sides:
        labels = Labels.load(_labels_path(cfg, recording.session, side))
        passed = labels.frames[labels.passed]
        frames |= set(rng.choice(passed, min(cfg.data.keyframes_per_session, len(passed)), replace=False).tolist())
    Keyframes.export(recording, sorted(frames), Path(cfg.paths.keyframes) / "images")


def validate(cfg: Config) -> None:
    recording = _recording(cfg)
    validator, annotations = LabelValidator(Labeller(cfg)), _annotations(cfg)
    directory = Path(cfg.paths.results) / f"validate-{recording.session}"
    for side in cfg.fit.sides:
        labels = Labels.load(_labels_path(cfg, recording.session, side))
        layers = {
            "lovo": validator.leave_one_view_out(recording, labels),
            "keyframes": validator.keyframes(recording, labels, annotations) if annotations else [],
            "manus": validator.manus(recording, labels),
        }
        for layer, rows in layers.items():
            if rows:
                metric = "error_px" if layer == "lovo" else "pa_mpjpe"
                _write_csv(rows, directory / f"{side}_{layer}.csv")
                print(f"{side} {layer}: median {metric} {np.nanmedian([row[metric] for row in rows]):.1f}")


def build_a(cfg: Config) -> None:
    sessions = cfg.data.sessions
    recordings = [Recording(Path(cfg.paths.recordings), session) for session in sessions]
    paths = [_labels_path(cfg, session, side) for session in sessions for side in cfg.fit.sides]
    labels = [Labels.load(path) for path in paths if path.exists()]
    print(ArmA(cfg).build(recordings, labels, _annotations(cfg)))


def build_b(cfg: Config) -> None:
    print(ArmB(cfg).build())


def train(cfg: Config) -> None:
    """Both arms: same model, number of images, augmentation, optimiser, schedule and early stopping.
    Only the training curves go to wandb; the checkpoints stay local."""
    pl.seed_everything(cfg.seed, workers=True)
    t = cfg.train
    if cfg.arm == "a":
        data, dataset = Path(cfg.paths.shards) / "arm_a" / f"fold{cfg.fold}", f"arm-a-f{cfg.fold}"
    else:  # Arm B does not depend on the fold: trained once per seed, tested on every fold
        data, dataset = Path(cfg.paths.shards) / "arm_b", "arm-b"
    name = f"train-{dataset}-s{cfg.seed}"
    augment = partial(random_example, rotation_deg=t.rotation_deg, scale_jitter=t.scale_jitter)
    train_loader = DataLoader(
        read_shards(data / "train", shuffle=True).map(augment), batch_size=t.batch_size, num_workers=t.workers
    )
    val_loader = DataLoader(read_shards(data / "val").map(to_example), batch_size=t.batch_size, num_workers=t.workers)
    logger = pl.loggers.WandbLogger(name=name, project=cfg.wandb.project, mode=cfg.wandb.mode, config=asdict(cfg))
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            Path(cfg.paths.checkpoints) / name, monitor="val/pa_mpjpe", save_weights_only=True
        ),
        pl.callbacks.EarlyStopping("val/pa_mpjpe", patience=t.patience),
    ]
    trainer = pl.Trainer(
        max_steps=t.max_steps,
        val_check_interval=t.val_every,
        check_val_every_n_epoch=None,
        precision=t.precision,  # type: ignore[arg-type]
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(HandPoseModel(t, cfg.paths), train_loader, val_loader)


def evaluate(cfg: Config) -> None:
    """Per-crop errors (<test set>.csv) and their summary (summary.json) for each test set of the fold."""
    evaluator = Evaluator(cfg)
    name = Path(cfg.adapter).parent.name if cfg.adapter else f"zero-shot-{cfg.model}"
    directory = Path(cfg.paths.results) / f"evaluate-{name}-f{cfg.fold}"
    directory.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for test_set in sorted((Path(cfg.paths.shards) / "test" / f"fold{cfg.fold}").iterdir()):
        rows = evaluator.evaluate(test_set)
        if rows:
            _write_csv(rows, directory / f"{test_set.name}.csv")
            summaries[test_set.name] = summarise(rows)
    (directory / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


def _recording(cfg: Config) -> Recording:
    if cfg.session is None:
        sys.exit("set session=<id>")
    return Recording(Path(cfg.paths.recordings), cfg.session)


def _labels_path(cfg: Config, session: str, side: str) -> Path:
    return Path(cfg.paths.labels) / f"{session}_{side}.npz"


def _annotations(cfg: Config) -> Keyframes | None:
    path = Path(cfg.paths.keyframes) / "annotations.json"
    return Keyframes.load(path) if path.exists() else None


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


STAGES: dict[str, Callable[[Config], None]] = {
    "label": label,
    "keyframes": keyframes,
    "validate": validate,
    "build-a": build_a,
    "build-b": build_b,
    "train": train,
    "evaluate": evaluate,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        sys.exit(__doc__)
    STAGES[sys.argv[1]](load_config(sys.argv[2:]))
