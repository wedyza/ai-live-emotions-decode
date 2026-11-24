# Five-point alignment experiment

This update adds preprocessing, NOT five extra numeric inputs to EfficientNet. The RGB model architecture stays unchanged. YuNet returns right eye, left eye, nose tip, right mouth corner, left mouth corner in full-frame pixel coordinates. Face messages now optionally carry landmarks and det_score. Old bbox-only messages still parse. The existing bundled ONNX weights and metadata are unchanged and continue to use bbox cropping.

## Alignment contract

common.face_preprocessing.prepare_face is shared by the notebook and ai-service. BBOX preserves the existing crop. yunet5_similarity_v1 fits a least-squares similarity transform (rotation, uniform scale, translation, no reflection) from all five points to a fixed custom template on a 224x224 RGB canvas. Template coordinates (fractions of 223): (.32,.38),(.68,.38),(.50,.56),(.37,.75),(.63,.75). OpenCV linear interpolation, replicated borders, then the model's existing PIL resize and ImageNet normalization.

This is a versioned initial template, not a validated optimum for emotion classification. Degenerate/nonfinite landmarks are rejected. Missing landmarks for an aligned model are logged and skipped; there is no hidden fallback to bbox crops. An aligned ONNX must be retrained/evaluated and exported with crop=yunet5_similarity_v1. Do not change the metadata on the existing weights just to activate alignment.

## 1. Build a separate dataset

From the repository root after installing this update:

```bash
uv sync --all-packages --no-dev --python 3.12 --locked
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python tools/build_landmarks.py --data datasets/faces --out datasets/faces_landmarks
```

Replace --data with the real existing dataset path. No source data is edited. Output must be a separate empty directory (not inside input). Source labels.csv needs id,file,label,x,y,width,height,split,subject_id,clip_id; classes.json and full-frame images must exist. Existing y is the bbox coordinate, not the class target.

The tool re-runs the SAME local YuNet on the saved full-frame JPEGs. It accepts only one detected face with bbox IoU >= .5 to the existing annotation. Invalid landmarks are rejected. This avoids arbitrarily selecting a different face. Check landmark_rejected.csv and landmark_report.json. A changed acceptance rate may change class balance; do not ignore it.

Accepted records retain IDs, labels, subjects, clips and split assignments. New bbox coordinates replace the bbox fields; source_bbox_* retains the previous ones. The output includes full-frame images, aligned PNG previews in crops/, JSON landmarks and explicit point coordinate columns, score and point order. No random resplitting occurs. Images are copied, so allocate extra disk space. The saved crops are for inspection; training recomputes the same alignment from full frames and points.

## 2. Inspect points and aligned faces before training

```bash
uv run --no-sync python tools/preview_landmarks.py --data datasets/faces_landmarks --out landmark_preview.jpg
```

Inspect eye/nose/mouth locations, full face coverage and tilted/extreme/occluded examples. Failures of landmarks or the inherited video labels are not repaired by alignment. This tool deliberately does not guess emotion labels.

## 3. Paired validation experiment

Install the optional training dependency group as described in README. Open notebooks/02_train_eval.ipynb. The notebook has no saved outputs. In a new first code cell, BEFORE configuration:

```python
import os
os.environ['EMO_DATA_DIR'] = '/your/dataset/faces_landmarks'
os.environ['EMO_FACE_PREPROCESSING'] = 'bbox without margin or alignment'
```

Run a bbox baseline on this retained dataset. Then run a separate experiment from original pretrained weights with EMO_FACE_PREPROCESSING='yunet5_similarity_v1'. Configuration, loaders and model must be rebuilt (prefer a fresh kernel between experiments). Both experiments must use the SAME retained IDs, subject splits, model initialization, epochs and augmentation settings. Never compare new aligned metrics to historical bbox metrics on a different set of accepted rows.

Choose preprocessing using validation only. Notebook now evaluates baseline test only at the final evaluation stage, after validation selection. Existing test results have already been inspected; for a defensible final improvement claim, reserve new independent held-out subjects. This update does not create such subjects or claim general improvement. Two full runs (2 head + 3 full epochs) have now been performed on the included data; see VERIFICATION_RU.md and reports/landmark_validation.

## 4. Inference only after an aligned model is trained and checked

The new notebook export writes the preprocessing mode in model.onnx.json and checks numerical agreement. Keep candidate weights separate from models/emotion_experiment until reviewed. To test a candidate locally set AI_EMOTION_MODEL=experimental and AI_EMOTION_MODEL_PATH to the candidate file in your own .env (or launch environment), restart ai-service and cv-service. Models with bbox metadata keep bbox preprocessing even though Face messages now contain landmarks. Don't run old consumers or rely on old queued bbox-only messages when validating an aligned candidate; upload a new short video.

Update checksums/model card only if you deliberately replace bundled weights. Existing bundled model is not changed by this update. There is no Git commit or push in any provided installer/command.

## Validation of this update

14 tests pass on Linux: original seven portability checks plus backward-compatible messages; exact YuNet point ordering; recovery of a known similarity transform; unchanged bbox pixels; invalid landmark rejection; training/inference alignment identity; dataset enrichment preserving labels, split and input. The dataset test uses a controlled detector fixture, not a claim about real facial accuracy. Existing real YuNet and ONNX smoke checks pass. The full 3613-frame dataset was rebuilt and both training modes were executed. Mac/Windows checks and Docker queue integration were not run in this environment.
