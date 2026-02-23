#!/usr/bin/env python3
"""Generate a self-contained TACOT material YAML for MFEM ablation case 1."""

from __future__ import annotations

import argparse
import math
import pathlib
import re
from typing import Dict, List, Tuple


FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_numeric_rows(path: pathlib.Path, min_cols: int) -> List[List[float]]:
    rows: List[List[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        toks = line.replace("\t", " ").split()
        vals: List[float] = []
        ok = True
        for t in toks:
            try:
                vals.append(float(t))
            except ValueError:
                ok = False
                break
        if ok and len(vals) >= min_cols:
            rows.append(vals)
    return rows


def parse_constant_properties(path: pathlib.Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    lines = path.read_text().splitlines()

    scalar_patterns = {
        "R": r"^\s*R\s+R\s+\[[^\]]*\]\s+(" + FLOAT_RE + r")\s*;",
        "K_v": r"^\s*K_v\s+K_v\s+\[[^\]]*\]\s+\((" + FLOAT_RE + r")",
        "K_c": r"^\s*K_c\s+K_c\s+\[[^\]]*\]\s+\((" + FLOAT_RE + r")",
        "eps_g_v": r"^\s*eps_g_v\s+eps_g_v\s+\[[^\]]*\]\s+(" + FLOAT_RE + r")\s*;",
        "eps_g_c": r"^\s*eps_g_c\s+eps_g_c\s+\[[^\]]*\]\s+(" + FLOAT_RE + r")\s*;",
        "rhoI1": r"^\s*rhoI\[1\]\s+rhoI\[1\]\s+\[[^\]]*\]\s+(" + FLOAT_RE + r")\s*;",
        "rhoI2": r"^\s*rhoI\[2\]\s+rhoI\[2\]\s+\[[^\]]*\]\s+(" + FLOAT_RE + r")\s*;",
        "epsI1": r"^\s*epsI\[1\]\s+epsI\[1\]\s+\[[^\]]*\]\s+(" + FLOAT_RE + r")\s*;",
        "epsI2": r"^\s*epsI\[2\]\s+epsI\[2\]\s+\[[^\]]*\]\s+(" + FLOAT_RE + r")\s*;",
    }

    for key, pat in scalar_patterns.items():
        rx = re.compile(pat)
        for ln in lines:
            m = rx.match(ln)
            if m:
                out[key] = float(m.group(1))
                break
        if key not in out:
            raise RuntimeError(f"Failed to parse {key} from {path}")

    # Parse reaction entries for phase 2.
    reactions: Dict[int, Dict[str, float]] = {}
    rkeys = ["F", "A", "E", "m", "n", "T", "h"]
    for ln in lines:
        line = ln.split("//", 1)[0].strip()
        if not line:
            continue
        for rk in rkeys:
            m = re.match(rf"^{rk}\[2\]\[(\d+)\].*?({FLOAT_RE})\s*;", line)
            if m:
                idx = int(m.group(1))
                reactions.setdefault(idx, {})
                reactions[idx][rk] = float(m.group(2))

    if not reactions:
        raise RuntimeError(f"No reactions parsed from {path}")

    ordered: List[Dict[str, float]] = []
    for idx in sorted(reactions):
        r = reactions[idx]
        missing = [k for k in rkeys if k not in r]
        if missing:
            raise RuntimeError(f"Reaction {idx} missing keys {missing}")
        ordered.append(
            {
                "F": r["F"],
                "A": r["A"],
                "E": r["E"],
                "m": r["m"],
                "n": r["n"],
                "T_threshold": r["T"],
                "h": r["h"],
            }
        )

    out["reactions"] = ordered  # type: ignore[index]
    return out


def group_by_pressure(rows: List[List[float]], keep_cols: Tuple[int, ...]) -> List[Tuple[float, List[List[float]]]]:
    grouped: Dict[float, List[List[float]]] = {}
    for row in rows:
        p = row[0]
        vals = [row[i] for i in keep_cols]
        grouped.setdefault(p, []).append(vals)

    out: List[Tuple[float, List[List[float]]]] = []
    for p in sorted(grouped.keys()):
        pts = sorted(grouped[p], key=lambda r: r[0])
        out.append((p, pts))
    return out


def fmt(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        raise ValueError("Non-finite value encountered in material data")
    return f"{x:.12g}"


def write_yaml(
    out_path: pathlib.Path,
    source_dir: pathlib.Path,
    constants: Dict[str, float],
    virgin: List[Tuple[float, List[List[float]]]],
    char: List[Tuple[float, List[List[float]]]],
    gas: List[Tuple[float, List[List[float]]]],
) -> None:
    lines: List[str] = []
    lines.append("material_name: TACOT_case1")
    lines.append("source:")
    lines.append(f"  path: {source_dir}")
    lines.append("constants:")
    lines.append(f"  R: {fmt(constants['R'])}")
    lines.append("phases:")
    lines.append(f"  rhoI: [{fmt(constants['rhoI1'])}, {fmt(constants['rhoI2'])}]")
    lines.append(f"  epsI: [{fmt(constants['epsI1'])}, {fmt(constants['epsI2'])}]")
    lines.append("transport:")
    lines.append(f"  K_v: {fmt(constants['K_v'])}")
    lines.append(f"  K_c: {fmt(constants['K_c'])}")
    lines.append(f"  eps_g_v: {fmt(constants['eps_g_v'])}")
    lines.append(f"  eps_g_c: {fmt(constants['eps_g_c'])}")
    lines.append("reactions:")
    for i, reac in enumerate(constants["reactions"], start=1):  # type: ignore[index]
        lines.append(f"  - id: r{i}")
        lines.append(f"    F: {fmt(reac['F'])}")
        lines.append(f"    A: {fmt(reac['A'])}")
        lines.append(f"    E: {fmt(reac['E'])}")
        lines.append(f"    m: {fmt(reac['m'])}")
        lines.append(f"    n: {fmt(reac['n'])}")
        lines.append(f"    T_threshold: {fmt(reac['T_threshold'])}")
        lines.append(f"    h: {fmt(reac['h'])}")

    def emit_table(name: str, grouped: List[Tuple[float, List[List[float]]]], header: str) -> None:
        lines.append(f"tables:")
        lines.append(f"  {name}:")
        lines.append(f"    columns: {header}")
        lines.append("    pressure_tables:")
        for p, rows in grouped:
            lines.append(f"      - p: {fmt(p)}")
            lines.append("        rows:")
            for r in rows:
                vals = ", ".join(fmt(v) for v in r)
                lines.append(f"          - [{vals}]")

    # Emit as a single 'tables' mapping to keep YAML tidy.
    lines.append("tables:")

    lines.append("  virgin:")
    lines.append("    columns: [T, cp, h, k]")
    lines.append("    pressure_tables:")
    for p, rows in virgin:
        lines.append(f"      - p: {fmt(p)}")
        lines.append("        rows:")
        for r in rows:
            lines.append(f"          - [{', '.join(fmt(v) for v in r)}]")

    lines.append("  char:")
    lines.append("    columns: [T, cp, h, k]")
    lines.append("    pressure_tables:")
    for p, rows in char:
        lines.append(f"      - p: {fmt(p)}")
        lines.append("        rows:")
        for r in rows:
            lines.append(f"          - [{', '.join(fmt(v) for v in r)}]")

    lines.append("  gas:")
    lines.append("    columns: [T, M, h, mu]")
    lines.append("    pressure_tables:")
    for p, rows in gas:
        lines.append(f"      - p: {fmt(p)}")
        lines.append("        rows:")
        for r in rows:
            lines.append(f"          - [{', '.join(fmt(v) for v in r)}]")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default="/home/quinnchr/Downloads/pato-3.1/data/Materials/Composites/TACOT",
        help="Path to TACOT folder containing constantProperties, virgin, char, gasProperties",
    )
    parser.add_argument(
        "--out",
        default="Input/material_tacot_case1.yaml",
        help="Output YAML path",
    )
    args = parser.parse_args()

    source_dir = pathlib.Path(args.source_dir).expanduser().resolve()
    out_path = pathlib.Path(args.out).resolve()

    const_path = source_dir / "constantProperties"
    virgin_path = source_dir / "virgin"
    char_path = source_dir / "char"
    gas_path = source_dir / "gasProperties"

    for p in [const_path, virgin_path, char_path, gas_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    constants = parse_constant_properties(const_path)

    virgin_rows = parse_numeric_rows(virgin_path, min_cols=7)
    char_rows = parse_numeric_rows(char_path, min_cols=7)
    gas_rows = parse_numeric_rows(gas_path, min_cols=5)

    virgin = group_by_pressure(virgin_rows, keep_cols=(1, 2, 3, 4))
    char = group_by_pressure(char_rows, keep_cols=(1, 2, 3, 4))
    gas = group_by_pressure(gas_rows, keep_cols=(1, 2, 3, 4))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(out_path, source_dir, constants, virgin, char, gas)

    print(f"Wrote {out_path}")
    print(f"Virgin pressure tables: {len(virgin)}")
    print(f"Char pressure tables: {len(char)}")
    print(f"Gas pressure tables: {len(gas)}")


if __name__ == "__main__":
    main()
