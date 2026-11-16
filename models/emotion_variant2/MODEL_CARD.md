# Variant 2 — image plus landmark inputs

Frozen HSEmotion EfficientNet-B0 backbone and frozen last image classifier from models/emotion_consensus (Variant1 comparison reference). Actual YuNet5 coordinates are eye-frame normalized (yunet5_eye_frame_v1), standardized by train statistics inside ONNX, processed by MLP10→32→16 and concatenated with1280 photo features. Only geometry encoder/projection trained. Inputs image float32[1,3,224,224] and landmarks float32[1,10]. Missing/invalid geometry is rejected; never replaced by zeros/template.

FER+ same development roles; geometry branch trained on3279 valid examples. Held-out750 images,744 detections, five common classes. Raw accuracy88.17% vsVariant1 88.44%; paired CI includes0. Selective accuracy91.68% at89.73% coverage vs91.39%/89.87%. Constant-coordinate control and shuffled geometry did not establish an advantage of actual geometry. Variant2 is implemented/trained, not recommended as a proven upgrade. Default unchanged.

Actual coordinate-only perturbations change logits. Numerical export checked20 images/40 views, max difference5.25e-6.36 tests. Unknown subject/pretraining overlap, no rare-class proof, no Docker or Mac/Windows acceptance. Not a diagnosis of mood. See VARIANTS_COMPARISON_RU.md and reports/variants for protocols, limits and provenance. Keep ONNX, metadata and policy together.
