"""Manual corpus operations. Publication and rollback require explicit confirmations."""

import argparse
import json
import sys
from pathlib import Path

from app.corpus.models import Release, Validation
from app.corpus.operations import bootstrap, publish, rollback, stage
from app.corpus.paths import INCOMING, PROCESSED, safe_id, safe_path
from app.corpus.storage import read_json, read_state, verify_release
from app.corpus.validation import validate


def summary(path, state):
    release = Release.model_validate(read_json(path / "release.json"))
    result = release.model_dump(exclude={"inventory"})
    result.update(
        location=path.parent.name,
        current=state.current_release == release.release_id,
        previously_published=release.release_id in state.published_releases,
        integrity_valid=False,
        publishable=False,
        validation_errors=[],
    )
    try:
        verify_release(path, require_sealed=False)
        result["integrity_valid"] = True
        result["publishable"] = (
            release.sealed
            and release.base_release == state.current_release
            and release.base_generation == state.generation
            and release.release_id not in state.published_releases
        )
        if (path / "validation.json").exists():
            report = Validation.model_validate(read_json(path / "validation.json"))
            result["validation_errors"] = report.errors
    except (OSError, ValueError) as exc:
        result["validation_errors"] = [str(exc)]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Controlled corpus releases")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--processed-root", type=Path, default=PROCESSED)
    common.add_argument("--incoming-root", type=Path, default=INCOMING)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("bootstrap", "stage", "validate", "publish", "rollback", "show", "list"):
        command = commands.add_parser(name, parents=[common])
        if name != "list":
            command.add_argument("--release", required=True)
        if name == "stage":
            command.add_argument("--package", type=Path, action="append", required=True)
        if name in ("publish", "rollback"):
            command.add_argument("--expected-current", required=True)
            command.add_argument("--confirm", required=True)
        if name == "publish":
            command.add_argument("--validation-sha", required=True)
    args = parser.parse_args(argv)
    try:
        root = args.processed_root
        if args.command == "bootstrap":
            result = bootstrap(args.release, root=root)
        elif args.command == "stage":
            result = stage(args.release, args.package, root=root, incoming_root=args.incoming_root)
        elif args.command == "validate":
            result, token = validate(args.release, root=root, incoming_root=args.incoming_root)
            print(f"publishable={str(result.publishable).lower()} validation_sha={token}")
            if not result.publishable:
                print("\n".join(result.errors), file=sys.stderr)
                return 1
            return 0
        elif args.command == "publish":
            result = publish(
                args.release, args.expected_current, args.validation_sha, args.confirm, root=root
            )
        elif args.command == "rollback":
            result = rollback(args.release, args.expected_current, args.confirm, root=root)
        elif args.command == "list":
            state = read_state(root)
            releases = []
            for kind in ("releases", "staging"):
                directory = safe_path(root / kind, root)
                if directory.exists():
                    for path in sorted(directory.iterdir()):
                        safe_path(path, root)
                        if (path / "release.json").is_file():
                            releases.append(summary(path, state))
            print(
                json.dumps(
                    {"state": state.model_dump(), "releases": releases},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        else:
            safe_id(args.release)
            candidates = [
                safe_path(root / kind / args.release, root) for kind in ("staging", "releases")
            ]
            existing = [p for p in candidates if p.exists()]
            if len(existing) != 1:
                raise ValueError("release missing or ambiguous")
            print(json.dumps(summary(existing[0], read_state(root)), ensure_ascii=False, indent=2))
            return 0
        data = result.model_dump()
        data.pop("inventory", None)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if args.command in ("bootstrap", "publish", "rollback"):
            print(
                "Publication completed. Restart/reload the backend process to adopt "
                f"release {args.release}."
            )
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
