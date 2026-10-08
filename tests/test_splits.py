"""Split invariants.

Every reported number depends on these holding.  The original submission sliced
``os.listdir()`` into five chunks with no shuffle and no seed, which made folds
both non-reproducible and non-stratified.
"""

from __future__ import annotations

import numpy as np
import pytest

from dit.data.splits import (
    leave_one_site_out,
    site_stratified_kfold_indices,
    split_indices,
    stratified_kfold_indices,
)


class TestStratifiedKfold:
    def test_disjoint_and_exhaustive(self) -> None:
        y = np.array([0] * 30 + [1] * 30 + [2] * 30)
        folds = list(stratified_kfold_indices(y, 5, seed=1))
        assert len(folds) == 5
        seen: list[int] = []
        for train, test in folds:
            assert np.intersect1d(train, test).size == 0, "fold overlap"
            seen.extend(test.tolist())
        assert sorted(seen) == list(range(y.size)), "every subject must be tested once"

    def test_class_proportions_are_preserved(self) -> None:
        y = np.array([0] * 40 + [1] * 40)
        folds = list(stratified_kfold_indices(y, 5, seed=2))
        for train, test in folds:
            test_labels = y[test]
            counts = np.bincount(test_labels, minlength=2)
            assert counts[0] >= 1 and counts[1] >= 1
            ratio = counts[0] / counts.sum()
            assert abs(ratio - 0.5) <= 0.15

    def test_is_deterministic_for_a_seed(self) -> None:
        y = np.array([0] * 25 + [1] * 25 + [2] * 25)
        first = [tuple(test.tolist()) for _, test in stratified_kfold_indices(y, 5, seed=7)]
        second = [tuple(test.tolist()) for _, test in stratified_kfold_indices(y, 5, seed=7)]
        third = [tuple(test.tolist()) for _, test in stratified_kfold_indices(y, 5, seed=8)]
        assert first == second
        assert first != third

    def test_seed_is_load_bearing(self) -> None:
        y = np.array([0] * 30 + [1] * 30)
        a = [tuple(test.tolist()) for _, test in stratified_kfold_indices(y, 5, seed=1)]
        b = [tuple(test.tolist()) for _, test in stratified_kfold_indices(y, 5, seed=2)]
        assert a != b

    @pytest.mark.parametrize("n_splits", [1, 0, -1])
    def test_invalid_split_count_raises(self, n_splits: int) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            list(stratified_kfold_indices(np.zeros(10, dtype=int), n_splits))

    def test_empty_labels_raise(self) -> None:
        with pytest.raises(ValueError, match="y cannot be empty"):
            list(stratified_kfold_indices(np.array([])))

    def test_split_count_cannot_exceed_samples_per_class(self) -> None:
        """A fold missing a class is uninterpretable, so it must not be yielded."""

        y = np.array([0, 0, 1, 1, 2, 2])
        with pytest.raises(ValueError, match="smallest class count"):
            list(stratified_kfold_indices(y, 5, seed=1))
        folds = list(stratified_kfold_indices(y, 2, seed=1))
        assert len(folds) == 2
        for _, test in folds:
            assert len(set(y[test].tolist())) == 3

    def test_every_fold_contains_every_class(self) -> None:
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        for _, test in stratified_kfold_indices(y, 5, seed=1):
            assert set(y[test].tolist()) == {0, 1, 2}


class TestLeaveOneSiteOut:
    def test_every_site_is_held_out_exactly_once(self) -> None:
        site = np.array([0] * 10 + [1] * 10 + [2] * 10)
        folds = list(leave_one_site_out(site))
        assert len(folds) == 3
        held_out = sorted(int(remaining) for _, _, remaining in folds)
        assert held_out == [0, 1, 2]

    def test_held_out_site_never_leaks_into_training(self) -> None:
        site = np.array([0] * 8 + [1] * 8 + [2] * 8 + [3] * 8)
        for train, test, held_out in leave_one_site_out(site):
            assert (site[test] == held_out).all()
            assert (site[train] == held_out).sum() == 0

    def test_disjoint_and_exhaustive(self) -> None:
        site = np.array([0] * 9 + [1] * 9 + [2] * 9)
        seen: list[int] = []
        for train, test, _ in leave_one_site_out(site):
            assert np.intersect1d(train, test).size == 0
            seen.extend(test.tolist())
        assert sorted(seen) == list(range(site.size))

    def test_sites_are_returned_in_stable_order(self) -> None:
        site = np.array([2] * 5 + [0] * 5 + [1] * 5)
        held_out = [remaining for _, _, remaining in leave_one_site_out(site)]
        assert held_out == sorted(held_out, key=str)

    def test_missing_sites_raise(self) -> None:
        with pytest.raises(ValueError, match="-1/missing"):
            list(leave_one_site_out(np.array([0, 0, -1, 1])))

    def test_empty_sites_raise(self) -> None:
        with pytest.raises(ValueError, match="site cannot be empty"):
            list(leave_one_site_out(np.array([])))

    def test_degenerate_single_site_yields_no_folds(self) -> None:
        # A single site has nothing to hold out against; the caller must notice.
        assert list(leave_one_site_out(np.array([0, 0, 0]))) == []


class TestSiteStratifiedKfold:
    def test_disjoint_and_exhaustive(self) -> None:
        site = np.array([0] * 30 + [1] * 30 + [2] * 30)
        folds = list(site_stratified_kfold_indices(site, 5, seed=1))
        assert len(folds) == 5
        seen: list[int] = []
        for train, test in folds:
            assert np.intersect1d(train, test).size == 0, "fold overlap"
            seen.extend(test.tolist())
        assert sorted(seen) == list(range(site.size)), "every subject must be tested once"

    def test_every_fold_holds_out_every_site(self) -> None:
        site = np.array([0] * 25 + [1] * 25 + [2] * 25)
        for _, test in site_stratified_kfold_indices(site, 5, seed=1):
            assert set(site[test].tolist()) == {0, 1, 2}

    def test_site_proportions_are_balanced(self) -> None:
        site = np.array([0] * 40 + [1] * 20)
        for _, test in site_stratified_kfold_indices(site, 5, seed=2):
            counts = np.bincount(site[test], minlength=2)
            assert counts[0] in (8, 9) and counts[1] in (4, 5)

    def test_is_deterministic_for_a_seed(self) -> None:
        site = np.array([0] * 25 + [1] * 25 + [2] * 25)
        first = [tuple(test.tolist()) for _, test in site_stratified_kfold_indices(site, 5, seed=7)]
        second = [tuple(test.tolist()) for _, test in site_stratified_kfold_indices(site, 5, seed=7)]
        third = [tuple(test.tolist()) for _, test in site_stratified_kfold_indices(site, 5, seed=8)]
        assert first == second
        assert first != third

    @pytest.mark.parametrize("n_splits", [1, 0, -1])
    def test_invalid_split_count_raises(self, n_splits: int) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            list(site_stratified_kfold_indices(np.zeros(10, dtype=int), n_splits))

    def test_split_count_cannot_exceed_smallest_site(self) -> None:
        site = np.array([0] * 3 + [1] * 20)
        with pytest.raises(ValueError, match="smallest site count"):
            list(site_stratified_kfold_indices(site, 5, seed=1))

    def test_missing_sites_raise(self) -> None:
        with pytest.raises(ValueError, match="-1/missing"):
            list(site_stratified_kfold_indices(np.array([0, 0, -1, 1])))

    def test_empty_sites_raise(self) -> None:
        with pytest.raises(ValueError, match="site cannot be empty"):
            list(site_stratified_kfold_indices(np.array([])))


class TestUnifiedSplitInterface:
    def test_stratified_strategy_delegates(self) -> None:
        y = np.array([0] * 20 + [1] * 20)
        folds = list(split_indices(y, None, strategy="stratified", n_splits=4, seed=1))
        assert len(folds) == 4

    def test_competition_alias_is_stratified(self) -> None:
        y = np.array([0] * 20 + [1] * 20)
        a = [tuple(test.tolist()) for _, test, _ in split_indices(y, None, "competition", 4, 3)]
        b = [tuple(test.tolist()) for _, test, _ in split_indices(y, None, "stratified", 4, 3)]
        assert a == b

    def test_loso_requires_site_metadata(self) -> None:
        with pytest.raises(ValueError, match="site metadata is required"):
            list(split_indices(np.array([0, 1, 0, 1]), None, strategy="loso"))

    @pytest.mark.parametrize("strategy", ["random", "group_kfold", ""])
    def test_unknown_strategy_raises(self, strategy: str) -> None:
        with pytest.raises(ValueError, match="unknown split strategy"):
            list(split_indices(np.array([0, 1, 0, 1]), None, strategy=strategy))

    def test_loso_reports_the_held_out_site(self) -> None:
        site = np.array([0] * 6 + [1] * 6)
        y = np.array([0, 1] * 6)
        folds = list(split_indices(y, site, strategy="loso"))
        assert [int(f[2]) for f in folds] == [0, 1]

    def test_site_stratified_strategy_delegates(self) -> None:
        site = np.array([0] * 20 + [1] * 20)
        y = np.array([0, 1] * 20)
        folds = list(split_indices(y, site, strategy="site_stratified", n_splits=4, seed=1))
        assert len(folds) == 4
        assert [int(f[2]) for f in folds] == [0, 1, 2, 3]

    def test_site_stratified_requires_site_metadata(self) -> None:
        with pytest.raises(ValueError, match="site metadata is required"):
            list(split_indices(np.array([0, 1, 0, 1]), None, strategy="site_stratified"))
