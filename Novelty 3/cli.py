"""Foreground commands for the Novelty 3 workflow."""

import argparse
import importlib.metadata
import json
import logging
import platform
import sys

from core.storage import ROOT


def environment():
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "rdkit", "torch", "torch-geometric", "xgboost"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    result = {"python": sys.version.split()[0], "executable": sys.executable,
              "platform": platform.platform(), "workspace": str(ROOT), "packages": packages}
    print(json.dumps(result, indent=2))
    return 0 if all(version != "not installed" for version in packages.values()) else 1


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def sensitivity(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return number


def add_device(parser):
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="Auto uses CUDA when available")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Novelty 3 documented-metabolite screening")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("environment", help="Inspect the current interpreter and dependencies")
    data = commands.add_parser("data", help="Acquire and audit research evidence")
    actions = data.add_subparsers(dest="action", required=True)
    actions.add_parser("fetch", help="Download or verify pinned public source snapshots")
    actions.add_parser("audit", help="Audit labels, identities and documented metabolic families")
    actions.add_parser("prepare", help="Build molecular graph cache and endpoint partitions")
    train = commands.add_parser("train", help="Fit or resume one endpoint's GINE models")
    calibrate = commands.add_parser("calibrate", help="Fit ensemble calibration and choose an endpoint threshold")
    for endpoint_command in (train, calibrate):
        endpoint_command.add_argument("endpoint", choices=("bbb", "pgp"))
        endpoint_command.add_argument("--seeds", type=positive_int, nargs="+", metavar="N",
                                      help="Default: 11 23 37 53 71. Match between training and calibration.")
        add_device(endpoint_command)
    train.add_argument("--epochs", type=positive_int, default=200)
    train.add_argument("--patience", type=positive_int, default=20)
    train.add_argument("--batch-size", type=positive_int, default=32)
    calibrate.add_argument("--min-sensitivity", type=sensitivity, default=0.90)
    screen = commands.add_parser("screen", help="Run sequential BBB then P-gp inference")
    explain = commands.add_parser("explain", help="Explain the decisions and bond sensitivity of a documented pair")
    explain.add_argument("--parent", required=True, metavar="INCHIKEY")
    explain.add_argument("--metabolite", required=True, metavar="INCHIKEY")
    benchmark = commands.add_parser("explain-benchmark", help="Benchmark bond sensitivity on held-out groups")
    add_device(explain)
    add_device(benchmark)
    evaluate = commands.add_parser("evaluate", help="Screen and evaluate on held-out data")
    validate = commands.add_parser("validate", help="Compare frozen models with curated paired experiments")
    commands.add_parser("fit-pgp-baseline", help="Train and calibrate the fingerprint P-gp comparator")
    pgp_compare = commands.add_parser("compare-pgp", help="Compare the trained P-gp GNN and fingerprint models")
    metabolite = commands.add_parser("pgp-metabolites", help="Validate P-gp cutoffs on new measured metabolites")
    metabolite_actions = metabolite.add_subparsers(dest="action", required=True)
    metabolite_actions.add_parser("audit", help="Review evidence and cohort overlap")
    metabolite_actions.add_parser("calibrate", help="Lock cutoffs using the development cohort")
    metabolite_actions.add_parser("evaluate", help="Compare locked cutoffs on new test groups")
    add_device(screen)
    add_device(evaluate)
    add_device(validate)
    add_device(pgp_compare)
    evaluate.add_argument("--budgets", nargs="+", type=positive_int, default=[5, 10, 20, 50])
    validate.add_argument("--budgets", nargs="+", type=positive_int, default=[1, 2, 3])
    args = parser.parse_args(argv)
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir / ("phase1.log" if args.command == "data" and args.action != "prepare"
                                           else "workflow.log"), level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Command %s", vars(args))
    try:
        if args.command == "environment":
            return environment()
        if args.command == "data":
            if args.action == "fetch":
                from core.sources import fetch_sources
                records = fetch_sources()
                print(f"Verified {len(records)} source snapshots. Run 'data audit' next.")
            elif args.action == "audit":
                from core.audit import run_audit
                return run_audit()
            else:
                from core.dataset import prepare
                prepare()
        elif args.command in ("train", "calibrate"):
            from core.learning import SEEDS, train, calibrate
            seeds = tuple(sorted(set(args.seeds))) if args.seeds else SEEDS
            if any(seed >= 2**63 - 1_000_000 for seed in seeds):
                raise ValueError("Training seeds must be smaller than 2^63 minus 1,000,000.")
            if args.command == "train":
                train(args.endpoint, seeds, args.epochs, args.patience, args.batch_size, args.device)
            else:
                calibrate(args.endpoint, seeds, args.min_sensitivity, args.device)
        elif args.command == "validate":
            from core.validation import validate_pairs
            validate_pairs(args.device, sorted(set(args.budgets)))
        elif args.command == "explain":
            from core.explanations import explain_pair
            explain_pair(args.parent, args.metabolite, args.device)
        elif args.command == "explain-benchmark":
            from core.explanation_benchmark import run_benchmark
            run_benchmark(args.device)
        elif args.command == "fit-pgp-baseline":
            from core.pgp_baseline import fit_baseline
            fit_baseline()
        elif args.command == "compare-pgp":
            from core.pgp_baseline import compare_pgp
            compare_pgp(args.device)
        elif args.command == "pgp-metabolites":
            from core.pgp_metabolites import audit_cohort, fit_cutoffs, evaluate_cutoffs
            {"audit": audit_cohort, "calibrate": fit_cutoffs, "evaluate": evaluate_cutoffs}[args.action]()
        else:
            from core.screening import run_pipeline
            budgets = sorted(set(args.budgets)) if args.command == "evaluate" else None
            run_pipeline(args.device, budgets)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as error:
        logging.exception("Command failed")
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logging.info("Command interrupted")
        print("Interrupted. Rerun the same command to resume from the last saved epoch or verified source snapshot.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
