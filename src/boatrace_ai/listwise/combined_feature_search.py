from __future__ import annotations

import argparse
import json

from ..db import connection, init_db
from .feature_search import FeatureVariants
from .feature_search import build_parser as build_feature_search_parser
from .feature_search import search


COMBINED_FEATURE_VARIANTS: FeatureVariants = (
    ("drop_base_pastlog", ("base_pastlog",)),
    (
        "keep_card_identity_context",
        ("card_numeric", "card_relative", "research_correlates"),
    ),
    (
        "keep_card_numeric",
        ("card_identity_context", "card_relative", "research_correlates"),
    ),
    (
        "keep_card_numeric_without_raw_equipment_ids",
        (
            "card_identity_context",
            "card_relative",
            "raw_equipment_identifiers",
            "research_correlates",
        ),
    ),
    (
        "keep_card_relative",
        ("card_identity_context", "card_numeric", "research_correlates"),
    ),
    (
        "drop_base_pastlog_research_correlates",
        ("base_pastlog", "research_correlates"),
    ),
    (
        "drop_research_correlates_rolling_history",
        ("research_correlates", "rolling_history"),
    ),
    (
        "drop_base_pastlog_rolling_history",
        ("base_pastlog", "rolling_history"),
    ),
)


def parse_combined_feature_variants(value: str) -> FeatureVariants:
    available = dict(COMBINED_FEATURE_VARIANTS)
    names = tuple(dict.fromkeys(
        item.strip() for item in value.split(",") if item.strip()
    ))
    if not names or any(name not in available for name in names):
        raise argparse.ArgumentTypeError("unsupported combined feature variants")
    return tuple((name, available[name]) for name in names)


def build_parser() -> argparse.ArgumentParser:
    parser = build_feature_search_parser()
    parser.description = "Fixed combined feature-group ablation search."
    parser.set_defaults(
        output="data/models/listwise_combined_feature_search_v1.json",
        cache_dir="data/models/listwise_combined_search_cache",
    )
    parser.add_argument(
        "--combined-feature-variants",
        type=parse_combined_feature_variants,
        default=None,
        help="Comma-separated registered combined feature variants.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    init_db(args.db)
    with connection(args.db) as conn:
        result = search(
            conn,
            args=args,
            variants=args.combined_feature_variants or COMBINED_FEATURE_VARIANTS,
        )
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"search_results", "daily"}
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
