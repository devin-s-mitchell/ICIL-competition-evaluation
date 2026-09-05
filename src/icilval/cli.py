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


def cmd_pools(args) -> int:
    import logging

    from .pools.build import (
        finalize,
        open_pool,
        stage_base,
        stage_composition,
        stage_environment,
        stage_object,
        stage_spatial,
        summary,
        verify_pool,
    )
    from .pools.schema import Pool
    from .pools.sources import Sources
    from .spec import _repo_root

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    spec = _spec(args)
    if args.pools_cmd == "build":
        out = Path(args.out)
        src = Sources.default(_repo_root() or Path.cwd())
        if args.libero_root:
            src.libero_root = Path(args.libero_root)
        if args.raw:
            raw = Path(args.raw)
            src.libero_datasets, src.libero_pro = raw / "libero_datasets", raw / "libero_pro"
            src.gen_goal_chain, src.gen_spatial_combination = (
                raw / "libero_gen_goal_chain",
                raw / "libero_gen_spatial_combination",
            )
        missing = src.check()
        if missing:
            print("missing sources:", *missing, sep="\n  ")
            return 1
        pool = open_pool(out, spec, args.version or str(spec.pools["version"]))
        stages = args.stage or [
            "base",
            "spatial",
            "environment",
            "object",
            "composition",
            "finalize",
        ]
        suites = tuple(args.suites) if args.suites else None
        kw = {"limit": args.limit, "validate": not args.no_validate}
        for stage in stages:
            if stage == "base":
                stage_base(
                    pool, spec, src, **({"suites": suites} if suites else {}), limit=args.limit
                )
            elif stage == "spatial":
                stage_spatial(pool, spec, src, **({"suites": suites} if suites else {}), **kw)
            elif stage == "environment":
                stage_environment(pool, spec, src, **({"suites": suites} if suites else {}), **kw)
            elif stage == "object":
                stage_object(pool, spec, src, **kw)
            elif stage == "composition":
                stage_composition(pool, spec, src, **kw)
            elif stage == "finalize":
                print("eligible:", finalize(pool, spec))
            else:
                print("unknown stage", stage)
                return 2
        pool.save()
        print(json.dumps(summary(pool), indent=1))
        return 0
    if args.pools_cmd == "verify":
        errors = verify_pool(Path(args.pool))
        for e in errors:
            print("error:", e)
        print(json.dumps(summary(Pool.load(args.pool)), indent=1))
        return 0 if not errors else 1
    if args.pools_cmd == "push":
        from .pools.hub import push_pool

        pool = Pool.load(args.pool)
        print(push_pool(Path(args.pool), args.repo or str(spec.pools["repo"]), pool.pool_version))
        return 0
    if args.pools_cmd == "pull":
        from .pools.hub import pull_pool

        print(
            pull_pool(
                args.repo or str(spec.pools["repo"]),
                args.version or str(spec.pools["version"]),
                Path(args.dest),
                revision=args.revision,
            )
        )
        return 0
    if args.pools_cmd == "generate":
        from .pools.build_gen import generate, import_generated
        from .spec import _repo_root as rr

        root = rr() or Path.cwd()
        bpp = Path(args.bpp_root) if args.bpp_root else root / "vendor" / "behavior_prompting"
        views = [f"libero_goal_{v}_view" for v in ("icil_object", "icil_chain")]
        run_dir = Path(args.run_dir)
        if not args.import_only:
            generate(
                bpp,
                root / "affordance.yaml",
                splits=["libero_goal"],
                views=views,
                suffix=args.suffix,
                run_dir=run_dir,
                n_demos=args.n_demos,
                workers=args.workers,
                python=args.python,
                dry_run=args.dry_run,
            )
        if args.pool and not args.dry_run:
            pool = Pool.load(args.pool)
            pool.pool_id = None
            got = import_generated(
                pool,
                spec,
                run_dir,
                views,
                axis_for_view={views[0]: "object", views[1]: "composition"},
                max_steps={"object": 400, "composition": 800},
                validate=not args.no_validate,
            )
            print("imported", got)
            print("eligible:", finalize(pool, spec))
        return 0
    return 2


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

    po = sub.add_parser("pools", help="build, verify and distribute evaluation pools")
    po_sub = po.add_subparsers(dest="pools_cmd", required=True)
    po_b = po_sub.add_parser("build")
    po_b.add_argument("--out", required=True)
    po_b.add_argument(
        "--stage",
        nargs="*",
        default=None,
        help="base spatial environment object composition finalize",
    )
    po_b.add_argument("--suites", nargs="*", default=None)
    po_b.add_argument(
        "--limit", type=int, default=None, help="tasks per suite/split (smoke builds)"
    )
    po_b.add_argument(
        "--no-validate", action="store_true", help="skip simulator validation (no instance lists)"
    )
    po_b.add_argument("--raw", default=None, help="raw cache dir (default ~/.cache/icilval/raw)")
    po_b.add_argument("--libero-root", default=None)
    po_b.add_argument("--version", default=None)
    po_v = po_sub.add_parser("verify")
    po_v.add_argument("pool")
    po_p = po_sub.add_parser("push")
    po_p.add_argument("pool")
    po_p.add_argument("--repo", default=None)
    po_l = po_sub.add_parser("pull")
    po_l.add_argument("--dest", required=True)
    po_l.add_argument("--repo", default=None)
    po_l.add_argument("--version", default=None)
    po_l.add_argument("--revision", default=None)
    po_g = po_sub.add_parser(
        "generate", help="run BPP's LIBERO-Gen scripts with affordance.yaml, then import"
    )
    po_g.add_argument("--run-dir", required=True)
    po_g.add_argument("--pool", default=None)
    po_g.add_argument("--suffix", default="icil")
    po_g.add_argument("--n-demos", type=int, default=12)
    po_g.add_argument("--workers", type=int, default=8)
    po_g.add_argument("--python", default="python")
    po_g.add_argument("--bpp-root", default=None)
    po_g.add_argument("--dry-run", action="store_true")
    po_g.add_argument("--import-only", action="store_true")
    po_g.add_argument("--no-validate", action="store_true")
    po.set_defaults(func=cmd_pools)

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
