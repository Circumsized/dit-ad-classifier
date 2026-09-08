"""Tract x node importance visualisation.

The publication convention for AFQ results is a tract-by-node heatmap per
diffusion metric: rows are the 18 bundles, columns the 100 sampled nodes,
colour the importance magnitude.  That single figure is the post-hoc
"estimated predictors" the competition asks for, and it is what lets a reader
check whether the model points at the bundles already reported in the
literature (left uncinate fasciculus, anterior thalamic radiation, cingulum)
or at something else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dit.data.layout import FeatureLayout


# Published discriminative AFQ regions for AD vs NC, as (tract-name substring,
# inclusive 1-based node interval).  These are the expected hits the post-hoc
# analysis checks the model against; a substring match keeps it robust to the
# exact tract naming a given AI4AD export uses.
LITERATURE_REGIONS: tuple[tuple[str, int, int], ...] = (
    ("uncinate", 75, 100),
    ("atr", 1, 13),
    ("thalamic", 1, 13),
    ("cingulum", 1, 10),
    ("callosum", 1, 10),
    ("cc", 1, 10),
)


def node_matrix(
    importance_rows: list[dict[str, object]],
    layout: FeatureLayout,
    *,
    metric_index: int | None = None,
) -> np.ndarray:
    """Reshape per-node importance into a [tracts, nodes] matrix.

    Rows and columns keep ``layout`` order so a figure can be compared
    tract-by-tract against the anatomical listing.  Metrics not requested are
    averaged out, which is appropriate when the ranking model was fit on all of
    them; pass ``metric_index`` to show one diffusion metric alone.
    """

    if layout.kind == "summary":
        raise ValueError(
            "summary features have no along-tract node axis; use a profile or "
            "metric view for the heatmap"
        )
    matrix = np.zeros((layout.n_tracts, layout.n_points), dtype=np.float64)
    if metric_index is None and layout.kind == "metric":
        metric_index = layout.metric_index
    for row in importance_rows:
        tract = int(row["tract_index"])
        if int(row["metric_index"]) != metric_index:
            continue
        node = row.get("node")
        if node is None:
            # summary-style rows cannot be placed on the node axis
            continue
        matrix[tract, int(node)] = float(row["magnitude"])
    return matrix


def render_node_heatmap(
    matrix: np.ndarray,
    tract_names: tuple[str, ...],
    *,
    title: str,
    ylabel: str = "tract",
    xlabel: str = "node",
    out_path: str | Path | None = None,
    figsize: tuple[float, float] = (6.0, 8.0),
    cmap: str = "viridis",
    show: bool = False,
) -> Path | None:
    """Render a heatmap and return the output path when one is given.

    Uses matplotlib's Agg backend so it works headless, which is how the
    evaluation pipeline runs.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(tract_names):
        raise ValueError("matrix rows must match tract_names")

    figure, axes = plt.subplots(figsize=figsize)
    axes.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")
    axes.set_yticks(range(len(tract_names)))
    axes.set_yticklabels(tract_names, fontsize=6)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title, fontsize=9)
    axes.set_xticks(np.linspace(0, max(values.shape[1] - 1, 1), 11))
    figure.colorbar(axes.images[0], ax=axes, shrink=0.8)
    figure.tight_layout()
    target: Path | None = None
    if out_path is not None:
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=150)
    if show:
        plt.show()
    plt.close(figure)
    return target


def render_metric_panel(
    importance_rows: list[dict[str, object]],
    layout: FeatureLayout,
    *,
    out_dir: str | Path,
    title_prefix: str = "AI4AD feature importance",
) -> list[Path]:
    """Render one heatmap per diffusion metric present in the ranking."""

    targets: list[Path] = []
    seen: set[int] = set()
    for row in importance_rows:
        metric = int(row["metric_index"])
        if metric in seen or metric >= layout.n_metrics:
            continue
        seen.add(metric)
        matrix = node_matrix(importance_rows, layout, metric_index=metric)
        if not np.any(matrix):
            continue
        title = f"{title_prefix} - {layout.metric_names[metric]}"
        target = render_node_heatmap(
            matrix,
            layout.tract_names,
            title=title,
            out_path=Path(out_dir) / f"importance_{layout.metric_names[metric]}.png",
        )
        if target is not None:
            targets.append(target)
    return targets


def literature_hits(
    importance_rows: list[dict[str, object]],
    layout: FeatureLayout,
) -> dict[str, object]:
    """Report whether the model's strongest nodes fall in published regions.

    For every tract whose name matches a :data:`LITERATURE_REGIONS` entry, the
    peak-importance node is compared against the expected interval.  A hit is
    evidence the model recovered a known biomarker; a miss is reported too, so
    the analysis is honest either way.  This is descriptive, not a claim about
    real-data performance.
    """

    if layout.kind == "summary":
        return {"status": "unavailable", "reason": "summary view has no node axis", "regions": []}

    # Peak node per tract (across whatever metrics the ranking used).
    peak: dict[int, tuple[int, float]] = {}
    for row in importance_rows:
        node = row.get("node")
        if node is None:
            continue
        tract = int(row["tract_index"])
        magnitude = float(row["magnitude"])
        if tract not in peak or magnitude > peak[tract][1]:
            peak[tract] = (int(node), magnitude)

    regions: list[dict[str, object]] = []
    for tract_index, tract_name in enumerate(layout.tract_names):
        lowered = tract_name.lower()
        for substring, lo, hi in LITERATURE_REGIONS:
            if substring not in lowered:
                continue
            entry: dict[str, object] = {
                "tract": tract_name,
                "expected_node_range": [lo, hi],
                "region_key": substring,
            }
            if tract_index in peak:
                node, magnitude = peak[tract_index]
                one_based = node + 1
                entry["peak_node"] = one_based
                entry["peak_magnitude"] = magnitude
                entry["hit"] = bool(lo <= one_based <= hi)
            else:
                entry["peak_node"] = None
                entry["hit"] = False
            regions.append(entry)

    n_hits = sum(1 for region in regions if region.get("hit"))
    return {
        "status": "computed" if regions else "no_matching_tracts",
        "n_regions": len(regions),
        "n_hits": n_hits,
        "regions": regions,
    }
