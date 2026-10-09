"""The shards of both arms. Both arms crop, store and count images identically; only the label source differs."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gloved_hands.capture import Camera, Recording
from gloved_hands.config import Config
from gloved_hands.keyframes import Keyframes
from gloved_hands.labelling import Labels
from gloved_hands.mano import Mano
from gloved_hands.recolor import GloveRecolorer, read_rgb
from gloved_hands.samples import Sample, write_shards


class ArmA:
    """Self-labelled rig recordings, one leave-one-subject-out fold.

    Writes ``arm_a/fold<k>/{train,val}`` and the fold's test sets ``test/fold<k>/{subject,task,keyframes}``:
    the held-out subject (pseudo-labels), the held-out task of the other subjects, and the held-out subject's
    human-annotated keyframes. Training uses exactly ``data.n_images`` images, one random camera per instant.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def build(self, recordings: list[Recording], labels: list[Labels], keyframes: Keyframes | None) -> dict[str, int]:
        cfg, task = self.cfg, self.cfg.data.held_out_task
        test_subject = sorted({r.subject for r in recordings})[cfg.fold]
        train = [r.session for r in recordings if r.subject != test_subject and r.task != task]
        val = set(self.rng.choice(train, max(1, round(cfg.data.val_fraction * len(train))), replace=False))
        sessions = {
            "train": set(train) - val,
            "val": val,
            "subject": {r.session for r in recordings if r.subject == test_subject},
            "task": {r.session for r in recordings if r.subject != test_subject and r.task == task},
        }
        instants = {split: self._instants(labels, names) for split, names in sessions.items()}
        if len(instants["train"]) < cfg.data.n_images:
            raise ValueError(
                f"fold {cfg.fold} has {len(instants['train'])} training images, fewer than "
                f"data.n_images={cfg.data.n_images}: set data.n_images to the smallest fold's count"
            )
        pick = np.sort(self.rng.choice(len(instants["train"]), cfg.data.n_images, replace=False))  # read in order
        instants["train"] = [instants["train"][i] for i in pick]

        fold, shards = f"fold{cfg.fold}", Path(cfg.paths.shards)
        by_session = {r.session: r for r in recordings}
        counts = {}
        for split, items in instants.items():  # every split is rewritten, so none keeps samples of an earlier build
            directory = shards / ("arm_a" if split in ("train", "val") else "test") / fold / split
            counts[split] = write_shards((self._sample(by_session[h.session], h, row) for h, row in items), directory)
        tested = [r for r in recordings if r.subject == test_subject]
        counts["keyframes"] = write_shards(self._keyframes(tested, keyframes), shards / "test" / fold / "keyframes")
        return counts

    def _instants(self, labels: list[Labels], sessions: set[str]) -> list[tuple[Labels, int]]:
        """(labels, row) of the instants of these recordings that passed quality control."""
        return [
            (hand, row) for hand in labels if hand.session in sessions for row in np.flatnonzero(hand.passed).tolist()
        ]

    def _sample(self, recording: Recording, labels: Labels, row: int) -> Sample:
        """One random camera of a labelled instant."""
        v = int(self.rng.integers(len(recording.cameras)))
        camera, frame, params = recording.cameras[v], int(labels.frames[row]), labels.params
        mano = {"global_orient": params.global_orient[row], "hand_pose": params.hand_pose[row], "betas": params.betas}
        key = f"{recording.session}/{labels.side}/{camera.name}/{frame:06d}"
        meta = _meta(recording, camera, "rig")
        return Sample.crop(key, recording[frame].color[v], camera, labels.joints[row], labels.side, meta, mano)

    def _keyframes(self, recordings: list[Recording], keyframes: Keyframes | None) -> Iterator[Sample]:
        """Every camera of every keyframe, with the triangulated human annotation as the reference."""
        if keyframes is None:
            return
        for recording in recordings:
            for (frame, side), joints in keyframes.triangulate(recording).items():
                if np.isfinite(joints).all(-1).sum() < 4:
                    continue
                for camera, image in zip(recording.cameras, recording[frame].color):
                    key = f"{recording.session}/{side}/{camera.name}/{frame:06d}"
                    yield Sample.crop(key, image, camera, joints, side, _meta(recording, camera, "keyframes"))


class ArmB:
    """DexYCB (S1 split: unseen subjects) with the hand recoloured to a recorded glove: ``arm_b/{train,val}``.

    The 3D joints are recomputed from DexYCB's MANO fits with the same MANO layer as Arm A, because DexYCB's
    own joints use other fingertip vertices than HaMeR and WiLoR.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        gloves = Path(cfg.paths.gloves)
        self.gloves = {variant: GloveRecolorer(gloves / variant) for variant in cfg.data.glove_variants}
        self.mano = {side: Mano(cfg.paths.mano, side) for side in ("right", "left")}
        self.pca = {side: _pca_basis(cfg.paths.mano, side) for side in ("right", "left")}

    def build(self) -> dict[str, int]:
        from dex_ycb_toolkit.factory import get_dataset

        data, directory = self.cfg.data, Path(self.cfg.paths.shards) / "arm_b"
        n_val = max(1, round(data.val_fraction * data.n_images))
        return {
            "train": write_shards(self._samples(get_dataset("s1_train"), data.n_images), directory / "train"),
            "val": write_shards(self._samples(get_dataset("s1_val"), n_val), directory / "val"),
        }

    def _samples(self, dataset: Any, count: int) -> Iterator[Sample]:
        made = 0
        for index in self.rng.permutation(len(dataset)):
            item = dataset[int(index)]
            label = np.load(item["label_file"])
            if np.all(label["joint_3d"] == -1):  # no hand in this frame
                continue
            yield self._sample(item, label)
            made += 1
            if made == count:
                return
        raise ValueError(f"only {made} labelled DexYCB frames, {count} requested")

    def _sample(self, item: dict[str, Any], label: Any) -> Sample:
        side, pose, file = item["mano_side"], label["pose_m"][0], Path(item["color_file"])
        mean, basis = self.pca[side]
        mano = {
            "global_orient": pose[:3],
            "hand_pose": mean + pose[3:48] @ basis,
            "betas": np.asarray(item["mano_betas"]),
        }
        with torch.no_grad():  # global_orient, hand_pose, betas, transl
            batch = [torch.tensor(np.asarray(x)[None], dtype=torch.float32) for x in (*mano.values(), pose[48:51])]
            joints = self.mano[side](*batch)[1][0].double().numpy()
        variant = list(self.gloves)[self.rng.integers(len(self.gloves))]
        image = self.gloves[variant](read_rgb(file), label["seg"] == 255, self.rng)  # 255 marks hand pixels
        i = item["intrinsics"]
        K = np.array([[i["fx"], 0, i["ppx"]], [0, i["fy"], i["ppy"]], [0, 0, 1.0]])
        subject, sequence, serial = file.parts[-4:-1]
        meta = {"source": "dexycb", "subject": subject, "session": sequence, "camera": serial, "glove": variant}
        key = f"dexycb/{subject}/{sequence}/{serial}/{file.stem}"
        return Sample.crop(key, image, Camera("dexycb", K, np.eye(3), np.zeros(3)), joints, side, meta, mano)


def _meta(recording: Recording, camera: Camera, source: str) -> dict[str, str]:
    """Strings only, so that batches collate them into lists; the frame is part of the key."""
    return {
        "source": source,
        "subject": recording.subject,
        "task": recording.task,
        "session": recording.session,
        "camera": camera.name,
    }


def _pca_basis(model_dir: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Mean and basis that turn DexYCB's 45 MANO PCA coefficients (flat_hand_mean=False) into axis-angle."""
    import smplx

    mano = smplx.MANO(model_dir, is_rhand=side == "right", num_pca_comps=45, flat_hand_mean=False)
    return np.asarray(mano.hand_mean), np.asarray(mano.np_hand_components)
