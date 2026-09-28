# Changelog

## v3 - motion-compensated locked monitor

- Replaces the SSIM-conditioned envelope with the single fixed configuration `Gmc-full-1ch-incl`.
- Input is detector predictions only. No frames, no SSIM, no optical flow, no labels.
- Camera motion is estimated as the per-axis median displacement of matched box centres and compensated before the localisation term is scored. The original assignment is kept; there is no second matching.
- Matching is one-to-one (Hungarian assignment) with class gating and an IoU floor of 0.05. v1 used greedy matching.
- Each transition has an explicit status (`MATCHED`, `NO_MATCH`, `ONE_SIDED`, `EMPTY_PAIR`). Empty and one-sided pairs are no longer given an invented consistency value: `EMPTY_PAIR` is undefined (`NaN`), while `NO_MATCH` and `ONE_SIDED` are judged as alarm 1.0.
- Frames are ordered numerically by frame index, so `frame10` follows `frame2`.
- The prediction loader ignores macOS `._*` metadata files.
- Reference tests (exact numeric equality on 100 stored transitions) and an identity test.
- Removed: the SSIM computation, the quantile-regression consistency envelope and drift detection, and the dataset-construction / incremental-training scripts of v1. They remain available at tag `v1-ssim`.

## v1 - SSIM consistency envelope (tag `v1-ssim`)

- Prediction consistency from IoU and class agreement of greedily matched boxes in consecutive frames.
- Adaptive envelope: quantile regression of IoU on frame SSIM, used to flag abnormal drift.
- Consistency-based frame selection and an incremental retraining pipeline.
