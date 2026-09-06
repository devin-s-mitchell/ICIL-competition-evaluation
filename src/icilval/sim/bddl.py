"""BDDL as data: parse, query, rewrite, serialize. Pure Python, no simulator.

A BDDL file is one s-expression. We keep it as nested lists of string atoms
so a rewrite touches only what it means to and everything else round-trips.
LIBERO-PRO's extra `(:perturbation_config …)` block is read into a dict and
stripped before the file reaches LIBERO's own parser.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

Node = list  # nested lists of str atoms

_TOKEN = re.compile(r"\(|\)|[^\s()]+")


def parse(text: str) -> Node:
    tokens = _TOKEN.findall(text)
    pos = 0

    def read() -> Any:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            out: list = []
            while tokens[pos] != ")":
                out.append(read())
            pos += 1
            return out
        if tok == ")":
            raise ValueError("unexpected ')'")
        return tok

    tree = read()
    if pos != len(tokens):
        raise ValueError("trailing tokens after the problem definition")
    return tree


def serialize(node: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if not isinstance(node, list):
        return pad + str(node)
    if not node:
        return pad + "()"
    if all(not isinstance(x, list) for x in node):
        return pad + "(" + " ".join(str(x) for x in node) + ")"
    head = node[0]
    if isinstance(head, list):
        inner = "\n".join(serialize(x, indent + 1) for x in node)
        return pad + "(\n" + inner + "\n" + pad + ")"
    rest = "\n".join(serialize(x, indent + 1) for x in node[1:])
    return pad + "(" + str(head) + "\n" + rest + "\n" + pad + ")"


def load(path: str | Path) -> Node:
    return parse(Path(path).read_text())


def dump(tree: Node, path: str | Path) -> None:
    Path(path).write_text(serialize(tree) + "\n")


# ------------------------------------------------------------------ sections
def section(tree: Node, name: str) -> Node | None:
    for item in tree:
        if isinstance(item, list) and item and item[0] == name:
            return item
    return None


def set_section(tree: Node, sec: Node) -> None:
    for i, item in enumerate(tree):
        if isinstance(item, list) and item and item[0] == sec[0]:
            tree[i] = sec
            return
    tree.append(sec)


def remove_section(tree: Node, name: str) -> Node | None:
    for i, item in enumerate(tree):
        if isinstance(item, list) and item and item[0] == name:
            return tree.pop(i)
    return None


def problem_name(tree: Node) -> str:
    return str(tree[1][1])


def set_problem_name(tree: Node, name: str) -> None:
    tree[1][1] = name


def language(tree: Node) -> str:
    sec = section(tree, ":language")
    return " ".join(sec[1:]) if sec else ""


def _typed_list(sec: Node | None) -> dict[str, str]:
    """`a b - typeA c - typeB` -> {a: typeA, b: typeA, c: typeB}."""
    out: dict[str, str] = {}
    if not sec:
        return out
    pending: list[str] = []
    items = sec[1:]
    i = 0
    while i < len(items):
        tok = items[i]
        if tok == "-":
            typ = items[i + 1]
            for name in pending:
                out[name] = typ
            pending = []
            i += 2
            continue
        pending.append(tok)
        i += 1
    return out


def fixtures(tree: Node) -> dict[str, str]:
    return _typed_list(section(tree, ":fixtures"))


def objects(tree: Node) -> dict[str, str]:
    return _typed_list(section(tree, ":objects"))


def obj_of_interest(tree: Node) -> list[str]:
    sec = section(tree, ":obj_of_interest")
    return list(sec[1:]) if sec else []


def predicates(tree: Node, name: str) -> list[list[str]]:
    sec = section(tree, name)
    if not sec:
        return []
    body = sec[1:]
    if name == ":goal" and body and isinstance(body[0], list) and body[0] and body[0][0] == "And":
        body = body[0][1:]
    return [list(p) for p in body if isinstance(p, list)]


def init_predicates(tree: Node) -> list[list[str]]:
    return predicates(tree, ":init")


def goal_predicates(tree: Node) -> list[list[str]]:
    return predicates(tree, ":goal")


def regions(tree: Node) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    sec = section(tree, ":regions")
    if not sec:
        return out
    for reg in sec[1:]:
        name = reg[0]
        info: dict[str, Any] = {"target": None, "ranges": [], "yaw_rotation": None}
        for attr in reg[1:]:
            if attr[0] == ":target":
                info["target"] = attr[1]
            elif attr[0] == ":ranges":
                info["ranges"] = [[float(v) for v in r] for r in attr[1]]
            elif attr[0] == ":yaw_rotation":
                info["yaw_rotation"] = [[float(v) for v in r] for r in attr[1]]
        out[name] = info
    return out


def scene_properties(tree: Node) -> dict[str, str]:
    sec = section(tree, ":scene_properties")
    return {
        attr[0]: attr[1]
        for attr in (sec[1:] if sec else [])
        if isinstance(attr, list) and len(attr) == 2
    }


# ------------------------------------------------------------------ rewrites
def _fmt(v: float) -> str:
    return repr(float(v))


def shift_region(tree: Node, region: str, dx: float, dy: float) -> None:
    sec = section(tree, ":regions")
    if not sec:
        raise KeyError(region)
    for reg in sec[1:]:
        if reg[0] != region:
            continue
        for attr in reg[1:]:
            if attr[0] == ":ranges":
                attr[1] = [
                    [
                        _fmt(float(r[0]) + dx),
                        _fmt(float(r[1]) + dy),
                        _fmt(float(r[2]) + dx),
                        _fmt(float(r[3]) + dy),
                    ]
                    for r in attr[1]
                ]
        return
    raise KeyError(region)


def set_region_yaw(tree: Node, region: str, lo: float, hi: float) -> None:
    sec = section(tree, ":regions")
    for reg in sec[1:] if sec else []:
        if reg[0] != region:
            continue
        for attr in reg[1:]:
            if attr[0] == ":yaw_rotation":
                attr[1] = [[_fmt(lo), _fmt(hi)]]
                return
        reg.append([":yaw_rotation", [[_fmt(lo), _fmt(hi)]]])
        return
    raise KeyError(region)


def rename_symbol(tree: Node, old: str, new: str) -> int:
    """Rename an atom everywhere, including as a prefix of region ids (`main_table_x` -> `t_x`)."""
    count = 0

    def walk(node: Any) -> Any:
        nonlocal count
        if isinstance(node, list):
            return [walk(x) for x in node]
        if node == old:
            count += 1
            return new
        if isinstance(node, str) and node.startswith(old + "_"):
            count += 1
            return new + node[len(old) :]
        return node

    tree[:] = walk(tree)
    return count


def set_fixture_type(tree: Node, fixture: str, new_type: str) -> None:
    sec = section(tree, ":fixtures")
    if not sec:
        raise KeyError(fixture)
    items = sec[1:]
    for i, tok in enumerate(items):
        if tok == fixture:
            j = i
            while items[j] != "-":
                j += 1
            items[j + 1] = new_type
            sec[1:] = items
            return
    raise KeyError(fixture)


def set_scene_properties(
    tree: Node, floor_style: str | None = None, wall_style: str | None = None
) -> None:
    current = scene_properties(tree)
    if floor_style:
        current["floor_style"] = floor_style
    if wall_style:
        current["wall_style"] = wall_style
    set_section(tree, [":scene_properties", *[[k, v] for k, v in current.items()]])


def swap_init_regions(tree: Node, a: str, b: str) -> None:
    sec = section(tree, ":init")
    if not sec:
        raise KeyError(":init")
    pa = next((p for p in sec[1:] if isinstance(p, list) and len(p) == 3 and p[1] == a), None)
    pb = next((p for p in sec[1:] if isinstance(p, list) and len(p) == 3 and p[1] == b), None)
    if pa is None or pb is None:
        raise KeyError(f"{a} or {b} not placed in :init")
    pa[2], pb[2] = pb[2], pa[2]


def object_region(tree: Node, obj: str) -> str | None:
    for p in init_predicates(tree):
        if len(p) == 3 and p[1] == obj and p[0] in ("On", "In"):
            return p[2]
    return None


# ---------------------------------------------------- LIBERO-PRO perturbation block
def _coerce(tok: str) -> Any:
    if tok in ("true", "false"):
        return tok == "true"
    try:
        return int(tok) if re.fullmatch(r"-?\d+", tok) else float(tok)
    except ValueError:
        return tok


def _block_to_dict(node: Node) -> Any:
    if not isinstance(node, list):
        return _coerce(node)
    if node and isinstance(node[0], str) and node[0].startswith(":"):
        key = node[0][1:]
        rest = node[1:]
        if (
            all(
                isinstance(x, list) and x and isinstance(x[0], str) and x[0].startswith(":")
                for x in rest
            )
            and rest
        ):
            return {key: {k: v for x in rest for k, v in _block_to_dict(x).items()}}
        if len(rest) == 1:
            return {key: _block_to_dict(rest[0])}
        return {key: [_block_to_dict(x) for x in rest]}
    return [_block_to_dict(x) for x in node]


def perturbation_config(tree: Node) -> dict[str, Any] | None:
    sec = section(tree, ":perturbation_config")
    if not sec:
        return None
    return _block_to_dict(sec)["perturbation_config"]


def strip_perturbation_config(tree: Node) -> dict[str, Any] | None:
    cfg = perturbation_config(tree)
    remove_section(tree, ":perturbation_config")
    return cfg


def libero_pro_initial_pose(tree: Node) -> dict[str, Any] | None:
    """The `07_initial_pose_position_angle` case: {target, delta_xy, min_delta_norm, yaw}."""
    cfg = perturbation_config(tree)
    if not cfg:
        return None
    pose = cfg.get("initial_pose")
    if not pose or not pose.get("enabled"):
        return None
    return {
        "target": pose["target"],
        "delta_xy": [float(v) for v in pose["delta_xy"]],
        "min_delta_norm": float(pose.get("min_delta_norm", 0.0)),
        "yaw": float(pose.get("yaw", 0.0)),
    }


# ---------------------------------------------------- table swap (environment perturbation)
TABLES: dict[str, dict[str, str]] = {
    "main_table": {"problem": "LIBERO_Tabletop_Manipulation", "type": "table"},
    "kitchen_table": {"problem": "LIBERO_Kitchen_Tabletop_Manipulation", "type": "kitchen_table"},
    "living_room_table": {
        "problem": "LIBERO_Living_Room_Tabletop_Manipulation",
        "type": "living_room_table",
    },
    "study_table": {"problem": "LIBERO_Study_Tabletop_Manipulation", "type": "study_table"},
}


def table_of(tree: Node) -> str | None:
    for name, typ in fixtures(tree).items():
        if typ in {t["type"] for t in TABLES.values()}:
            return name
    return None


def swap_table(tree: Node, new_table: str) -> str:
    """Move the whole scene onto another LIBERO table; regions keep their table-relative ranges."""
    if new_table not in TABLES:
        raise KeyError(new_table)
    old = table_of(tree)
    if old is None:
        raise ValueError("no table fixture in this problem")
    if old != new_table:
        rename_symbol(tree, old, new_table)
        set_fixture_type(tree, new_table, TABLES[new_table]["type"])
    set_problem_name(tree, TABLES[new_table]["problem"])
    return old
