# Reliable expression classifier

Experimental EfficientNet-B0, HSEmotion enet_b0_8_best_afew backbone, head adapted on 4510 detected FER2013 train images with frozen backbone/BatchNorm and KL distillation. Seven expression classes have reference labels; contempt is abstained from. YuNet five-point alignment yunet5_similarity_v1 is required. RGB 224×224, ImageNet normalization, batch1 logits; original/flip averaged and calibrated by policy.json. Missing/inconsistent/low-scoring inputs produce uncertain.

Selected using PublicTest development partitions. Frozen before PrivateTest: 3464 images after exact deduplication. Reliable answered 1071: 977 label matches, 94 errors (91.22% among answers; coverage 30.92%). This is not 91% overall accuracy and not validation of every class. In particular, predicted disgust precision was 9/20, sadness 58/78. See RELIABILITY_REPORT_RU.md and reports/reliability for per-class failures, all logits, provenance and confidence intervals.

Local PyTorch/ONNX numerical checks pass. Full Docker and Windows/macOS acceptance untested here. Unknown pretraining overlap and no subject IDs. Weak/noisy reference labels, grayscale48×48 benchmark domain. Not an inference of internal mood or mental health. No guarantee on new videos.

Source HSEmotion: https://github.com/av-savchenko/face-emotion-recognition ; mirror/split SHA sources: reports/reliability/benchmark_sources.json. Retain model.onnx, metadata and policy together. Do not silently swap policy/weights/crop metadata. Original models retained as comparisons. No automatic Git actions.
