# 2020 upstream reference code (archived)

These files are the recovered 2020 upstream codebase that the original course
submission in the parent directory derives from. Archived verbatim on
2026-10-08 from a `main`-branch zip download; the timestamps inside the zip
read 2020-11-29/30. Same-origin evidence: `deep_model.Trans` is the earlier
variant of `../transformer.py`'s `Trans` class (both pair an in-model Softmax
with `nn.BCELoss`), and `data_loader_d.py` plays the role of the
`data_loader.py` that the course scripts import but which was never committed.

**Do not run.** Kept as a historical reference only; the maintained pipeline is
the `dit/` package. Beyond the defects the parent file records for the course
submission (Softmax feeding BCELoss, whole-row zeroing when a tract row
contains any NaN, unseeded shuffles), this code hardcodes `MCAD_AFQ_competition.mat`
at the working-directory root, hardcodes 20 tracts while the official data has
18, and its `TextCNN` model cannot even be constructed. `data_division.py`
shuffles without a seed, so its generated splits are one run's record, not a
reproducible protocol.

What was salvaged from it: the site-stratified fold protocol (each fold's test
slice draws a proportional chunk from every site) now lives as
`site_stratified_kfold_indices` in `dit/data/splits.py`, made deterministic and
covered by tests. `dataset_txt/` keeps the original five fold index lists as a
record of what this code once ran with.
