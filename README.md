# consistency — label-free, motion-compensated temporal consistency for video object detection

This package scores how consistent an object detector's outputs are between consecutive video frames, **without ground-truth labels**. A falling score is a runtime warning that the detector may be degrading, for example under domain drift.

It implements one fixed configuration, the **locked cell `Gmc-full-1ch-incl`**: camera motion is compensated before scoring, and the result is a single alarm value per frame pair.

## What goes in, what comes out

**Input:** detector predictions for two consecutive frames, each box as `[cls, cx, cy, w, h]` or `[cls, cx, cy, w, h, conf]` in normalised coordinates (YOLO `.txt` format; a missing `conf` counts as 1.0). No images, no optical flow, no labels.

**Output** per frame pair (`MonitorResult`):

| field | meaning |
|---|---|
| `status` | `MATCHED`, `NO_MATCH`, `ONE_SIDED` or `EMPTY_PAIR` |
| `N_t`, `N_t1` | number of boxes in frame *t* and *t+1* **after** per-class NMS |
| `K` | number of matched pairs |
| `G` | mean IoU of the matched pairs, before motion compensation (`NaN` unless `MATCHED`) |
| `G_mc` | the same pairs' mean IoU after shifting frame *t* by the estimated global shift (`NaN` unless `MATCHED`) |
| `dx`, `dy` | estimated global shift, in the units of the input coordinates (`0.0, 0.0` if there are no matches) |
| `motion_magnitude` | `hypot(dx, dy)` |
| `consistency` | `G_mc * K / max(N_t, N_t1, 1)` when `MATCHED`; `0.0` for `NO_MATCH` and `ONE_SIDED`; `NaN` for `EMPTY_PAIR` |
| `alarm` | `1 - consistency` when `MATCHED`; `1.0` for `NO_MATCH` and `ONE_SIDED`; `NaN` for `EMPTY_PAIR` |

`consistency` and `alarm` lie in [0, 1] up to floating-point rounding.

## How it works

1. **Class-wise non-maximum suppression** on each frame (IoU threshold 0.5, no confidence cut-off). `N_t` and `N_t1` are the counts that remain.
2. **Matching** between the two frames with the Hungarian algorithm on cost `1 - IoU`. Boxes of different classes cannot match, and pairs with IoU below 0.05 are rejected. (A spatial gate on centre distance also exists, but with normalised coordinates it never binds, so admission is decided by class and the IoU floor alone.)
3. **Global shift:** the median displacement of the matched box centres, taken separately for x and y: `(dx, dy)`.
4. **Compensation:** frame *t* boxes are shifted by `(dx, dy)` (translation only), and the IoU of the **same** matched pairs is recomputed. The mean of these IoUs is `G_mc`. No second matching is done.
5. **Score:** `consistency = G_mc * K / max(N_t, N_t1, 1)`, and `alarm = 1 - consistency`.

Status rules (counts are after NMS):
- `EMPTY_PAIR`: both frames have no boxes. `consistency` and `alarm` are `NaN`.
- `ONE_SIDED`: exactly one frame has no boxes. `alarm = 1.0`.
- `NO_MATCH`: both frames have boxes but no pair is admitted. `alarm = 1.0`.
- `MATCHED`: at least one pair is admitted.

Configuration name `Gmc-full-1ch-incl`:
- `Gmc`: localisation `G` is computed after motion compensation.
- `full`: every unmatched box counts in full, through the denominator `max(N_t, N_t1, 1)`.
- `1ch`: one composite alarm, rather than separate disappearance and appearance channels.
- `incl`: `ONE_SIDED` and `NO_MATCH` transitions are judged (alarm 1.0) rather than skipped. `EMPTY_PAIR` remains undefined.

## Quick start

```bash
pip install -r requirements.txt
python examples/run_on_yolo_txt.py --pred-dir path/to/yolo_txt_predictions
```

The script prints one CSV row per pair of consecutive frames (`--out FILE` writes to a file instead). Files are ordered numerically by frame index, and pairs are formed only within one sequence; see `consistency/io.py` for the accepted file-name patterns.

```python
from consistency.monitor import compute_locked_monitor
result = compute_locked_monitor(preds_t, preds_t1)
print(result.consistency, result.alarm)
```

## Label-free: what that covers

The **score** never reads labels. Turning the score into an alarm needs a **threshold**, and how you choose it decides whether the whole monitor is label-free. This package does not include any thresholding.
- **Label-free:** set the threshold from the score's distribution on data you consider healthy (for example, a low quantile of `consistency`).
- **Supervised:** fit the threshold on labelled degradations, for example to a target false-positive rate. The score is still label-free; the threshold is not.

## Cost

| platform | median per frame pair | what was timed |
|---|---|---|
| Jetson TX2 (Python 3.6.9, NumPy 1.19.5, SciPy 1.5.4, one BLAS thread) | 1.88 ms | the complete `compute_locked_monitor`, on 6,287 real transitions (median about 7 boxes per frame) |
| Apple-silicon laptop (macOS, Python 3.13) | 48 µs | the matching stage only (`compute_frame_pair_signal`, with NMS and Hungarian matching), on real transitions; the full monitor was not timed on this machine |

Timings exclude detector inference and file parsing. The per-pair input is a few kilobytes of Python lists (median 4.0 KB on the real transitions, 12 KB at 25 boxes per frame). The whole benchmark process on the TX2 peaked at about 100 MB resident memory, most of it the Python runtime. No image processing is involved. These are measurements from one run on each platform; timings on your hardware will differ.

## Limitations

- **Translation only:** rotation and zoom are not compensated.
- **Median shift:** the estimate degrades if more than half of the matched objects move differently from the camera.
- **Matching is done before compensation and not redone:** under a large camera shift, pairs whose IoU falls below 0.05 are never matched and cannot be recovered by the compensation.
- **Whole-frame gaps:** if a frame has no boxes at all (or no pair is admitted), the pair is `ONE_SIDED` or `NO_MATCH` and the alarm is 1.0, even if the detector is correct. Individual objects entering or leaving the frame lower `K / max(N_t, N_t1)` but the pair stays `MATCHED`.
- **Normalised coordinates:** the fixed spatial-gate constant assumes coordinates in [0, 1]. With pixel coordinates the gate would become active and change the result.

## Tests

```bash
pip install pytest
pytest tests/
```

- `test_reference.py` reproduces the stored reference signals of 100 transitions exactly (maximum absolute error 0.0).
- `test_identity.py` checks the identity anchor: with `G` in place of `G_mc` (and every unmatched box counted in full) the composite `G * K / max(N_t, N_t1, 1)` equals the uncompensated consistency `R = G * m` to machine precision. It also checks the status rules and that a pure global shift is compensated.

## History

- **v3 (current):** motion-compensated locked monitor `Gmc-full-1ch-incl`. No SSIM, no image input.
- **v1:** SSIM-conditioned consistency envelope. Superseded; available at tag [`v1-ssim`](https://github.com/Commin/consistency/tree/v1-ssim).

See [CHANGELOG.md](CHANGELOG.md).

## Citation

_To be added._

## License

_To be chosen._
