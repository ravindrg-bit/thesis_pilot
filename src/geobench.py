"""
geobench.py — load and work with GEO-Bench (V3 primary dataset; Aggarwal et al. 2024).

Kept reusable (not inline in a script) because the same loader feeds the pilot now
and the full Stage-4/5 collection later. GEO-Bench specifics from the feasibility test
are handled here: multi-config layout (configs include 'train'/'test'), record fields
query/tags/sources/sugg_idx, and NO native id field (so query_id is derived).
"""

import hashlib

from datasets import load_dataset, get_dataset_config_names, DatasetDict

import thesis_config as cfg


def derive_query_id(query_text: str) -> str:
    """Stable, content-derived id: prefix + first N hex of SHA-256(query text).

    Uses hashlib, NOT Python's built-in hash() (which is randomised per process):
    the same query must map to the same id on any machine, forever. This is what
    lets pilot ids and production ids line up and a re-download map back to the
    same canonical records.
    """
    digest = hashlib.sha256(query_text.strip().encode("utf-8")).hexdigest()
    return f"{cfg.QUERY_ID_PREFIX}_{digest[:cfg.QUERY_ID_HASH_LEN]}"


def load_geobench(prefer: str = None):
    """Load GEO-Bench, resolving its config/split with fallbacks rather than
    assuming a layout. Returns (dataset, provenance_dict)."""
    repo = cfg.GEOBENCH_REPO
    prefer = prefer or cfg.GEOBENCH_SPLIT  # 'test'

    try:
        configs = get_dataset_config_names(repo)
    except Exception:
        configs = []

    config = prefer if prefer in configs else (configs[0] if configs else None)
    loaded = load_dataset(repo, config) if config else load_dataset(repo)

    # `loaded` may be a Dataset or a DatasetDict -> reduce to one flat Dataset.
    if isinstance(loaded, DatasetDict):
        keys = list(loaded.keys())
        split = (prefer if prefer in keys
                 else "test" if "test" in keys
                 else "train" if "train" in keys
                 else keys[0])
        ds = loaded[split]
    else:
        split = config or "default"
        ds = loaded

    import datasets as _d
    info = getattr(ds, "info", None)
    provenance = {
        "geobench_repo": repo,
        "available_configs": configs,
        "resolved_config": config,
        "resolved_split": split,
        "datasets_lib_version": _d.__version__,
        "num_rows": getattr(ds, "num_rows", None),
        "dataset_version": str(info.version) if info and info.version else None,
        "dataset_license": getattr(info, "license", None) if info else None,
        "fingerprint": getattr(ds, "_fingerprint", None),
        "revision_pinned": None,  # pilot loads latest; pin a commit SHA for production
    }
    return ds, provenance


def profile_tags(ds, top: int = 15) -> dict:
    """Count tag values across the dataset (tags is a list per record), most common
    first. For understanding the data and designing stratified sampling at scale-up
    — NOT used for the pilot's random selection."""
    from collections import Counter
    c = Counter()
    for row in ds:
        tags = row.get("tags")
        if isinstance(tags, (list, tuple)):
            c.update(str(t) for t in tags)
        elif tags is not None:
            c.update([str(tags)])
    return dict(c.most_common(top))
