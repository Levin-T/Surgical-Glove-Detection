# gloved-hands

3D hand pose for gloved hands. WiLoR is fine-tuned with LoRA on two training sets of equal size and
compared on the same test sets:

- **Arm A**: rig recordings, labelled automatically by a multi-view MANO fit (HaMeR keypoints, SAM 2 glove
  masks, depth), with quality control.
- **Arm B**: DexYCB, with the bare hands recoloured to look like a recorded glove.

Everything runs locally. Training curves also go to Weights & Biases, so runs can be followed from anywhere.

## Install

```bash
pip install -e ".[dev]"
```

The models and their code come separately, from their own repositories and licences:

| What | Where it goes (defaults in `config.py`, root `$GLOVED_HANDS_DATA`) |
|---|---|
| [WiLoR](https://github.com/rolpotamias/WiLoR): clone it, `pip install -r requirements.txt` and add the clone to `PYTHONPATH` (it has no installer). Download `wilor_final.ckpt` and `detector.pt` as its README says; `model_config.yaml` is in its `pretrained_models/` | `models/wilor/` |
| [HaMeR](https://github.com/geopavlakos/hamer): `pip install -e .[all]` in a clone, then `hamer_ckpts/` from its `fetch_demo_data.sh` | `models/hamer/hamer_ckpts/` |
| [SAM 2](https://github.com/facebookresearch/sam2): `pip install -e .` in a clone; the weights load from Hugging Face | — |
| MANO: `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), and `mano_mean_params.npz` from WiLoR's `mano_data/` | `models/mano/` |
| Arm B only: [dex-ycb-toolkit](https://github.com/NVlabs/dex-ycb-toolkit) with `DEX_YCB_DIR` set | — |

## Connect the capture pipeline

`capture.Recording` is a stub. Implement it on top of the capture pipeline. It must provide:

- `session`, `subject` and `task`.
- `cameras`: a list of `Camera(name, K, R, t)`. A world point X in metres maps to `R @ X + t`.
- `len(recording)` and `recording[i]`, which returns a `Frame` with:
  - one undistorted RGB image per camera;
  - one depth map per camera, in metres and aligned to the colour image, with 0 meaning no depth;
  - optionally, MANUS joints per side, in the joint order of `mano.JOINT_NAMES`.

## Stages

Run `python -m gloved_hands <stage> key=value ...`. Any config field can be overridden on the command line, for example `train.lr=3e-4`.

```bash
export GLOVED_HANDS_DATA=/data/gloved-hands

python -m gloved_hands label session=S01_pinch_1        # MANO labels of both hands, with QC
python -m gloved_hands keyframes session=S01_pinch_1    # images of labelled frames, to annotate in CVAT
python -m gloved_hands validate session=S01_pinch_1     # leave-one-view-out, keyframes, MANUS (RQ2) -> results/

python -m gloved_hands build-a fold=0 "data.sessions=[S01_pinch_1,S02_pinch_1,...]" data.held_out_task=suture
python -m gloved_hands build-b
python -m gloved_hands train arm=a fold=0 seed=0
python -m gloved_hands train arm=b seed=0                # Arm B does not depend on the fold
python -m gloved_hands evaluate fold=0 adapter=<checkpoint>   # per-crop CSVs and summary.json -> results/
python -m gloved_hands evaluate fold=0 model=wilor       # zero-shot baselines: wilor, hamer
```

Notes:

- **Arm sizes.** `data.n_images` is the training set size of both arms. `build-a` fails if a fold has fewer images; in that case, set `data.n_images` to the smallest fold's count and build again. A build replaces the shards it writes.
- **Keyframes.** Set up CVAT as follows:
  1. Create one skeleton per hand. Its label name contains `left` or `right`, and its points are named as in `mano.JOINT_NAMES`.
  2. Upload `keyframes/images/` as a zip, so the paths `<session>/<camera>/<frame>.png` are kept.
  3. Mark occluded joints as not visible.
  4. Export as COCO Keypoints to `keyframes/annotations.json`.

## Data layout

```
$GLOVED_HANDS_DATA/
  recordings/                        whatever Recording reads
  labels/<session>_<side>.npz
  keyframes/images/, keyframes/annotations.json
  gloves/<variant>/<name>.png        the glove alone in the rig, with <name>_mask.png
  shards/arm_a/fold<k>/{train,val}/
  shards/arm_b/{train,val}/
  shards/test/fold<k>/{subject,task,keyframes}/
  checkpoints/train-<dataset>-s<seed>/   LoRA weights only
  results/                           CSVs of validate and evaluate, and evaluate's summary.json
  models/
```

Shards are webdataset tars of 100 crops of 256 × 256 pixels. Labels are stored in each crop's camera frame. Left hands are stored mirrored, as right hands.

## wandb

Only `train` uses wandb: it logs the losses, `val/pa_mpjpe` and the config, so a run can be followed from a phone. Checkpoints, labels, shards and results stay local. Set `wandb.mode=offline` on a machine without internet and upload later with `wandb sync`, or `wandb.mode=disabled` to turn it off.

## Development

```bash
ruff format && ruff check && mypy src
```
