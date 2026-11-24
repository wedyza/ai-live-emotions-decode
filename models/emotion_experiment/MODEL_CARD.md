# Experimental expression classifier

Model: EfficientNet-B0, initialized from enet_b0_8_best_afew. Selected checkpoint: head-only epoch 2. Eight output classes in model.onnx.json. This is expression classification, not a measurement of a person's internal emotional state. Softmax scores are not calibrated probabilities.

Inputs: uint8 RGB face cropped from a decoded full frame by x/y/width/height; no alignment or margin; PIL bilinear resize 224x224; divide by 255; ImageNet mean/std; NCHW float32, batch exactly 1. Output: eight logits. Use the accompanying metadata, not an assumed class order.

Dataset: 794 train / 183 validation / 310 test after filtering; split by source subject groups, duplicate heuristic applied. Labels inherited from whole videos, not verified frame-by-frame. Calm/confusion excluded. No contempt/fear/sadness ground-truth support. Cross-source identities not independently verified.

Test (before -> after): accuracy .3581 -> .5065; balanced accuracy .3685 -> .4628; macro-F1 .4076 -> .4497; weighted F1 .4106 -> .5144. Macro-F1 delta clip-bootstrap 95% CI [-.027,+.110] includes zero. Same-person clips can remain dependent; this bootstrap does not account for that. No established general superiority over baseline.

The user checked export on 10 images, all top classes equal; max absolute output difference about 8e-6. This package was independently checked for load and synthetic inference, not re-evaluated on the user's private dataset.

Weights are included so inference does not require the author's local files or dataset. Training data is not included. Source-model and dataset rights were not independently established by this packaging work; no new license grant is made here. See source author's project https://github.com/av-savchenko/hsemotion-onnx .

Hash: models/checksums.json. Do not replace weights without metadata, evaluation and updated checksums.
