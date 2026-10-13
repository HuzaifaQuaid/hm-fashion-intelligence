"""Command line interface: python -m hmfashion.cli <command>."""
from pathlib import Path
import argparse

from hmfashion.analytics import generate_report
from hmfashion.appdata import build_app_tables, build_deployment
from hmfashion.demo import generate_demo
from hmfashion.forecasting import train_forecast
from hmfashion.pipeline import build_database
from hmfashion.recommender import train_recommender
from hmfashion.repurchase import train_repurchase

ROOT = Path(__file__).resolve().parents[2]


def ingest(args):
    result = build_database(Path(args.raw_dir), Path(args.db), args.source)
    print(f"Ingested {result['source']} data into {args.db}")
    for check in result["checks"]:
        if check["count"]:
            print(f"  {check['check']}: {check['count']:,} ({check['severity']})")


def build(args):
    db, artifacts = Path(args.db), Path(args.artifacts)
    print(generate_report(db, artifacts)["purchase_events"], "purchase events")
    print("repurchase:", train_repurchase(db, artifacts)["metrics"]["logistic_regression"]["roc_auc"])
    print("recommender:", train_recommender(db, artifacts)["metrics"])
    print("forecast:", train_forecast(db, artifacts)["selected_method"])
    build_app_tables(db)
    print("Dashboard tables refreshed.")


def report(args):
    generate_report(Path(args.db), Path(args.artifacts))


def demo(args):
    generate_demo(Path(args.raw_dir))
    print(f"Synthetic demo written to {args.raw_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hmfashion")
    parser.add_argument("--db", default=str(ROOT / "data/hm.duckdb"))
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest"); p.add_argument("--raw-dir", default=str(ROOT / "data/raw")); p.add_argument("--source", default="hm"); p.set_defaults(func=ingest)
    sub.add_parser("build").set_defaults(func=build)
    sub.add_parser("report").set_defaults(func=report)
    p = sub.add_parser("demo"); p.add_argument("--raw-dir", default=str(ROOT / "data/demo_raw")); p.set_defaults(func=demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
