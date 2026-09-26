"""Run real local API integration separately from offline tests."""

import argparse
import json
import sys

from app.evaluation.api_generation_gate import load_v1_baseline, validate_generation
from app.evaluation.api_smoke import DEFAULT_OUTPUT, ROOT, load_cases, run
from app.evaluation.generation_cli import load_datasets


def main(argv=None) -> int:
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run")
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    safety = commands.add_parser("safety-probes")
    safety.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/processed/evaluation/task023b/safety"
    )
    validation = commands.add_parser("validate-generation")
    validation.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-generation":
            load_cases()  # Both frozen hashes and the independently reviewed specification.
            generation, retrieval = load_datasets(
                ROOT / "data/evaluation/generation_questions.json",
                ROOT / "data/evaluation/retrieval_questions.json",
            )
            result = validate_generation(
                json.loads(args.report.read_text(encoding="utf-8")),
                generation,
                retrieval,
                baseline=load_v1_baseline(),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["passed"] else 1
        report = run(args.output_dir, safety=args.command == "safety-probes")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"{report['verdict']} passed={report['pass_count']}/{len(report['case_ids'])}")
    if report["error"]:
        print(report["error"], file=sys.stderr)
    for case in report["case_results"]:
        print(
            f"{case['question_id']} passed={case['passed']} "
            f"seconds={case['elapsed_seconds']:.3f} error={case['error']}"
        )
    print(f"reports={args.output_dir}")
    return 0 if report["verdict"] == "api_e2e_smoke_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
