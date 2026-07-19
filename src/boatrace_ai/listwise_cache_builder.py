from __future__ import annotations

import argparse
import json
from pathlib import Path

from .db import connection, init_db
from .feature_tuning import load_complete_race_ids
from .listwise_feature_search import feature_variants, load_variant_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build one listwise feature-search cache.")
    parser.add_argument("--db", default="data/boatrace.sqlite")
    parser.add_argument("--cache-dir", default="data/models/listwise_search_cache")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--n-features", type=int, default=1 << 12)
    parser.add_argument("--batch-races", type=int, default=1_000)
    args = parser.parse_args(argv)
    variants = dict(feature_variants())
    if args.variant not in variants:
        raise ValueError(f"unknown variant: {args.variant}; choices: {', '.join(variants)}")
    init_db(args.db)
    with connection(args.db) as conn:
        race_keys = load_complete_race_ids(conn)
        dataset, source = load_variant_dataset(
            conn,
            race_keys=race_keys,
            cache_dir=Path(args.cache_dir),
            name=args.variant,
            dropped=variants[args.variant],
            n_features=args.n_features,
            batch_races=args.batch_races,
        )
    print(json.dumps({
        "variant": args.variant,
        "source": source,
        "races": dataset.race_count,
        "matrix_shape": list(dataset.matrix.shape),
        "matrix_nnz": int(dataset.matrix.nnz),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
