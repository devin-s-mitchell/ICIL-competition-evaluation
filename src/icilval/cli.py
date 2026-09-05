"""`icilval` command line. argparse only: it must import inside the minimal container."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _spec(args):
    from .spec import load_spec

    return load_spec(args.spec)


def cmd_spec(args) -> int:
    from .spec import load_spec, spec_path, validate_spec

    if args.spec_cmd == "fingerprint":
        print(load_spec(args.spec).fingerprint)
        return 0
    path = Path(args.spec) if args.spec else spec_path()
    errors = validate_spec(json.loads(path.read_text()))
    if errors:
        print("invalid:", ", ".join(errors))
        return 1
    print(f"{path}: ok")
    return 0


def cmd_keys(args) -> int:
    from .canon import Signer

    out = Path(args.out)
    secret = out / "validator.ed25519"
    if secret.exists() and not args.force:
        print(f"{secret} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    signer = Signer.generate()
    signer.save(secret)
    (out / "validator.pub").write_text(signer.verify_key_hex + "\n")
    print(signer.verify_key_hex)
    return 0


def cmd_store(args) -> int:
    from .canon import Signer
    from .store.verify import verify_store
    from .store.writer import Store

    spec = _spec(args)
    if args.store_cmd == "init":
        signer = Signer.from_file(args.key)
        store = Store(args.root, spec, signer)
        manifest = store.init(signer.verify_key_hex, args.pool_id)
        print(json.dumps(manifest, indent=2))
        return 0
    if args.store_cmd == "verify":
        report = verify_store(args.root, spec)
        for w in report.warnings:
            print("warning:", w)
        for e in report.errors:
            print("error:", e)
        print(
            f"records={report.records} events={report.events} media={report.media} {'OK' if report.ok else 'FAILED'}"
        )
        return 0 if report.ok else 1
    if args.store_cmd == "mirror":
        from .store.mirror import mirror_store

        n = mirror_store(args.root, args.repo, spec, message=args.message, all_files=args.all)
        print(f"mirrored {n} files to {args.repo}")
        return 0
    return 2


def cmd_queue(args) -> int:
    from .queue import Queue

    q = Queue(args.queue)
    if args.queue_cmd == "add":
        entry, pos = q.add(args.repo, args.revision, duel_size=args.duel_size, source="cli")
        print(f"{entry.ref.entry} key={entry.key} position={pos}")
        return 0
    if args.queue_cmd == "list":
        for i, e in enumerate(q.entries(), start=1):
            print(
                f"{i:3d} {e.key} {e.repo}@{e.revision} size={e.duel_size or '-'} accepted={e.accepted_at}"
            )
        ip = q.state.in_progress
        print(f"in_progress={ip.event_id if ip else '-'} block={q.block}")
        return 0
    if args.queue_cmd == "remove":
        print("removed" if q.remove(args.key) else "not found")
        return 0
    return 2


def cmd_admin(args) -> int:
    from .admin import AdminServer
    from .queue import Queue

    spec = _spec(args)
    key = Path(args.pub).read_text().strip() if args.pub else ""
    server = AdminServer(spec, Queue(args.queue), args.token, key, bind=args.bind, port=args.port)
    print(f"admin listening on http://{server.bind}:{server.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def cmd_units(args) -> int:
    from .ids import ModelRef, duel_id
    from .pools.schema import Pool
    from .pools.units import derive_units

    spec = _spec(args)
    pool = Pool.load(args.pool)
    ch_repo, ch_rev = args.challenger.split("@", 1)
    challenger = ModelRef.make(ch_repo, ch_rev)
    king = None
    if args.king:
        k_repo, k_rev = args.king.split("@", 1)
        king = ModelRef.make(k_repo, k_rev)
    did = duel_id(spec.version, spec.track_id, challenger, king)
    units = [u.as_dict() for u in derive_units(pool, spec, did, args.size)]
    doc = {"duel_id": did, "pool_id": pool.pool_id, "size": spec.size_of(args.size), "units": units}
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"{len(units)} units -> {args.out}")
    else:
        print(json.dumps(doc, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="icilval", description="ICIL competition validator")
    p.add_argument("--spec", help="path to spec.json (default: packaged / repo root)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("spec", help="inspect the contract")
    s.add_argument("spec_cmd", choices=["fingerprint", "validate"])
    s.set_defaults(func=cmd_spec)

    k = sub.add_parser("keys", help="generate the validator signing key")
    k.add_argument("keys_cmd", choices=["generate"])
    k.add_argument("--out", default="keys")
    k.add_argument("--force", action="store_true")
    k.set_defaults(func=cmd_keys)

    st = sub.add_parser("store", help="result store operations")
    st_sub = st.add_subparsers(dest="store_cmd", required=True)
    st_init = st_sub.add_parser("init")
    st_init.add_argument("root")
    st_init.add_argument("--key", required=True, help="validator.ed25519 secret file")
    st_init.add_argument("--pool-id", default=None)
    st_ver = st_sub.add_parser("verify")
    st_ver.add_argument("root")
    st_mir = st_sub.add_parser("mirror")
    st_mir.add_argument("root")
    st_mir.add_argument("--repo", required=True, help="Hugging Face dataset repo owner/name")
    st_mir.add_argument("--message", default="publish")
    st_mir.add_argument(
        "--all", action="store_true", help="upload every file, not only changed ones"
    )
    st.set_defaults(func=cmd_store)

    q = sub.add_parser("queue", help="challenger queue")
    q.add_argument("--queue", default="queue/queue.json")
    q_sub = q.add_subparsers(dest="queue_cmd", required=True)
    q_add = q_sub.add_parser("add")
    q_add.add_argument("repo")
    q_add.add_argument("revision")
    q_add.add_argument("--duel-size", default=None)
    q_sub.add_parser("list")
    q_rm = q_sub.add_parser("remove")
    q_rm.add_argument("key")
    q.set_defaults(func=cmd_queue)

    a = sub.add_parser("admin", help="submission intake server")
    a.add_argument("admin_cmd", choices=["serve"])
    a.add_argument("--queue", default="queue/queue.json")
    a.add_argument("--token", required=True)
    a.add_argument("--pub", default=None, help="validator.pub, echoed by /admin/health")
    a.add_argument("--bind", default=None)
    a.add_argument("--port", type=int, default=None)
    a.set_defaults(func=cmd_admin)

    u = sub.add_parser("units", help="derive a duel's unit list")
    u.add_argument("units_cmd", choices=["derive"])
    u.add_argument("--pool", required=True)
    u.add_argument("--challenger", required=True, help="repo@revision")
    u.add_argument("--king", default=None, help="repo@revision")
    u.add_argument("--size", default=None)
    u.add_argument("--out", default=None)
    u.set_defaults(func=cmd_units)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
