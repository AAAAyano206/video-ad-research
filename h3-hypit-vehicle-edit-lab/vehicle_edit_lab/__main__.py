import argparse
import sys
from pathlib import Path

from .conditioning import learn_direction, prepare_package
from .files import read_json, write_json
from .hypit import export_run
from .planning import get_job, make_plan
from .replay import replay


def job_from(args):
    return get_job(read_json(args.plan), args.job)


def run_plan(args):
    matrix = read_json(args.matrix)
    plan = make_plan(matrix, read_json(args.spec), args.phase, args.attribute)
    write_json(args.output, plan)
    print(f"{args.output}: {len(plan['jobs'])} jobs")


def run_learning(args):
    learn_direction(args.package_root, args.pair, args.nuisance, args.attribute, args.tensor_key, args.output)
    print(args.output)


def run_preparation(args):
    print(prepare_package(args.package_root, job_from(args), args.direction, args.alpha, args.tensor_key))


def run_replay(args):
    print(replay(
        job_from(args), args.package_root, args.bridge_script, args.bridge_root,
        args.worker_python, args.results_dir, args.timeout,
    ))


def run_export(args):
    export_run(args.author, args.output, args.target, args.clip, args.receipt, args.reuse)
    print(args.output)


def parser():
    root = argparse.ArgumentParser(description="Plan H3 vehicle edits and prepare Hypit runs.")
    commands = root.add_subparsers(required=True)
    plan = commands.add_parser("plan", help="expand shot and seed inputs into a phase plan")
    plan.add_argument("--matrix", type=Path, default=Path("experiments/matrix.json"))
    plan.add_argument("--spec", type=Path, required=True)
    plan.add_argument("--phase", choices=("pilot", "holdout", "continuity"), required=True)
    plan.add_argument("--attribute")
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(handler=run_plan)

    learn = commands.add_parser("learn-direction", help="learn a feature direction from donor pairs")
    learn.add_argument("--package-root", type=Path, required=True)
    learn.add_argument("--attribute", required=True)
    learn.add_argument("--tensor-key", default="t000000")
    learn.add_argument("--pair", nargs=2, metavar=("SOURCE", "TARGET"), action="append", required=True)
    learn.add_argument("--nuisance", type=Path, action="append", default=[])
    learn.add_argument("--output", type=Path, required=True)
    learn.set_defaults(handler=run_learning)

    prepare = commands.add_parser("prepare", help="create an A1 or A2 condition package")
    submit = commands.add_parser("replay", help="run A0, A1 or A2 with the existing H3 bridge")
    for command in (prepare, submit):
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--job", required=True)
        command.add_argument("--package-root", type=Path, required=True)
    prepare.add_argument("--direction", type=Path, required=True)
    prepare.add_argument("--alpha", type=float, required=True)
    prepare.add_argument("--tensor-key", default="t000000")
    prepare.set_defaults(handler=run_preparation)
    submit.add_argument("--bridge-script", type=Path, required=True)
    submit.add_argument("--bridge-root", type=Path, required=True)
    submit.add_argument("--worker-python", type=Path, required=True)
    submit.add_argument("--timeout", type=int, default=1800)
    submit.add_argument("--results-dir", type=Path, default=Path("results/runs"))
    submit.set_defaults(handler=run_replay)

    export = commands.add_parser("export-svrun", help="bind clips and prior results to an existing Hypit author source")
    export.add_argument("--author", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--target", default="final.video")
    export.add_argument("--clip", action="append", default=[], metavar="OUTPUT=MP4")
    export.add_argument("--receipt", action="append", default=[], metavar="OUTPUT=RECEIPT_JSON")
    export.add_argument("--reuse", action="append", default=[], metavar="OUTPUT=BUILD_ID:PREVIOUS_OUTPUT")
    export.set_defaults(handler=run_export)
    return root


def main():
    args = parser().parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
