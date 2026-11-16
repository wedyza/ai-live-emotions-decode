# Expanded FER+ candidate — NOT the default

Original frozen EfficientNet-B0 backbone with a head trained on soft FER+ votes. Policy v2, five-point yunet5_similarity_v1 alignment, original/flip agreement, per-class calibrated score thresholds. Supports anger/happiness/neutral/sadness/surprise under the policy; refuses fear/disgust/contempt due insufficient reliability evidence. This does NOT prevent misclassification of those true expressions into other classes.

On a newly reserved strong-consensus holdout (804 images, 796 detected): raw accuracy86.68% versus previous head76.76%. Expanded answered693 with627 label matches (90.48%, coverage86.19%). Previous reliable answered306 with299 matches (97.71%, coverage38.06%). New candidate failed the predeclared release gate requiring precision at least the old policy, so default remains previous reliable. Neutral precision78.79%, surprise88.16% on this holdout; do not call every supported class reliable.

Same dataset family, curated by >=6/10 annotator agreement. No subject IDs, unknown pretraining overlap, disgust only1 / contempt8 holdout examples. Not comparable directly to earlier FER2013 PrivateTest metrics. No test-time threshold changes. No psychological diagnosis or certainty of true emotions. Full details and reproducible scripts: CONSENSUS_REPORT_RU.md, reports/consensus.

Source annotations Microsoft/FERPlus, cite Barsoum et al. ICMI2016. Image/model usage terms must be checked independently. Runtime ONNX, metadata, policy must stay together. No automatic promotion to default or Git actions.
