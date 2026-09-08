# DO NOT RUN — legacy course-submission scripts

These five scripts are the **original course submission**, kept only as a
historical record. They are not maintained, not tested, and none of them can
produce a valid result. The maintained pipeline is the `dit/` package — see the
README and `docs/OPTIMIZATION_PLAN.md`.

| Script | Task | Status |
|---|---|---|
| `ML.py` | Random forest, binary | Broken: label bug F1 makes every label constant 0 |
| `transformer.py` | Transformer, binary | Broken: fatal defects F2/F3/F4 below |
| `transformer123.py` | Transformer, 3-class | Broken: fatal defects F2/F3/F4 below |
| `test.py` | Binary inference | Broken: ensemble logic F6, NaN handling F7 |
| `test123.py` | 3-class inference | Broken: depends on uncommitted `data_loader23.py` |

## Why they are unrunnable (verified 2026-09-06, see OPTIMIZATION_PLAN §2.1)

- **F1** `ML.py:25-26` — `label.index(max(label))` takes the *index* of the max,
  not the label value. Every subject gets label 0; `fit()` raises on a
  single-class vector.
- **F2** `transformer*.py:20` — `nn.TransformerEncoder(..., norm="T")` is not a
  valid argument; torch raises `TypeError` at construction.
- **F3** `transformer*.py:8` — imports `data_loader.py` / `data_loader23.py`,
  which were never committed; the `.mat`/`.npy` data files are also absent.
- **F4** `transformer*.py` — Softmax inside the model feeding `nn.BCELoss`,
  which expects logits; the loss is applied to a probability simplex.
- **F5** `transformer123.py:24` / `test123.py:84` — the task definition is
  self-contradictory: the 3-class head outputs three logits while the official
  primary task and README are binary AD vs NC, and `test123.py` then remaps with
  `prediction+1`. The same repository reports two incompatible tasks, so which
  result the author intended cannot be determined.
- **F6** `test.py:78-95` — ensemble "voting" adds max-prob to class 1 and
  *min*-prob to class 2, then divides by 3 unconditionally; ties favor AD.
- **F7** `test.py:53-54` — if any element of a row is NaN, the whole row's
  non-zero values are zeroed, destroying real measurements.
- **F8** shape contract hardcodes 20 tokens × 100 nodes; the official data has
  18 tracts, and nothing validates the rearrangement.

Two security/reproducibility patches were applied in place on 2026-09-07 before
retirement: `torch.load(..., weights_only=True)` in `test*.py` and seeded
`random.Random(42)` validation splits in `transformer*.py` (plan defects S4;
they were also the accepted-false-positive LOW CWE-330 findings in the Mimosa
scans — the seed *is* the fix). Those patches do not make the scripts runnable.

## Use the real pipeline instead

```bash
python -m dit.cli evaluate --synthetic --model linear_svm
python -m pytest
```
