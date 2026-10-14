#!/usr/bin/env python3
"""Derive the paper's tables and quoted numbers from collected runs.

Usage: paper_tables.py --out DIR RUN_DIR [RUN_DIR ...]

Each RUN_DIR is one directory written by collect.py.  If two directories
contain the same benchmark case, the one given later is used.  Speedups are
paired: the two cases of a ratio are taken from the same repetition of the
same run directory.  The script writes tab_*.tex and numbers.tex into DIR.
"""
import argparse
import json
import random
import statistics as st
from pathlib import Path

TBD = r"\tbd"


def load(run_dir):
    """Return {case: {repetition: median_ns}} for one run directory."""
    cases = {}
    for rep in sorted(Path(run_dir).glob("run-*")):
        if (rep / "FAILED").exists() or not (rep / "benchmarks.json").exists():
            continue
        for est in (rep / "criterion").glob("**/new/estimates.json"):
            case = str(est.relative_to(rep / "criterion").parent.parent)
            median = json.loads(est.read_text())["median"]["point_estimate"]
            cases.setdefault(case, {})[rep.name] = median
    return cases


class Data:
    def __init__(self, run_dirs):
        self.obs = {}       # case -> {repetition: ns}
        self.src = {}       # case -> run directory
        for d in run_dirs:
            for case, reps in load(d).items():
                self.obs[case] = reps
                self.src[case] = str(d)

    def med(self, case):
        return st.median(self.obs[case].values()) if case in self.obs else None

    def ratios(self, base, opt):
        if base not in self.obs or opt not in self.obs or self.src[base] != self.src[opt]:
            return None
        reps = sorted(set(self.obs[base]) & set(self.obs[opt]))
        return [self.obs[base][r] / self.obs[opt][r] for r in reps]

    def ratio(self, base, opt):
        rs = self.ratios(base, opt)
        return st.median(rs) if rs else None


def bootstrap(xs, level=0.99, n=20000):
    """Percentile-bootstrap confidence interval for the median."""
    rng = random.Random(1)
    meds = sorted(st.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    tail = (1 - level) / 2
    return meds[int(tail * n)], meds[int((1 - tail) * n) - 1]


def num(v):
    """Two decimals below ten, one decimal from ten up."""
    return TBD if v is None else (f"{v:.2f}" if v < 9.995 else f"{v:.1f}")


def ms(ns):
    return TBD if ns is None else num(ns / 1e6)


def x(r):
    return TBD if r is None else f"{r:.2f}"


def pct(r):
    return TBD if r is None else f"{(r - 1) * 100:.0f}\\%"


def share(r):
    """Fraction of the baseline time that a speedup r removes."""
    return (1 - 1 / r) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    d = Data(args.runs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    macros = {}

    # Binding (Table 2).
    c = "BN254_combined_"
    bf, bu, pf, pu = (c + s for s in ("boolean_full", "boolean_upper", "projective_full", "projective_upper"))
    bbf, pbf = c + "boolean_bf", c + "projective_bf"
    mul = ("BN254_chained_mul_standard", "BN254_chained_mul_upper_limb")
    (out / "tab_combined.tex").write_text(
        f"Boolean & {ms(d.med(bf))}\\,ms & {ms(d.med(bu))}\\,ms ($\\mathbf{{{x(d.ratio(bf, bu))}\\times}}$) \\\\\n"
        f"Projective & {ms(d.med(pf))}\\,ms ($\\mathbf{{{x(d.ratio(bf, pf))}\\times}}$)"
        f" & {ms(d.med(pu))}\\,ms ($\\mathbf{{{x(d.ratio(bf, pu))}\\times}}$) \\\\\n")
    macros["numMulSpeedup"] = x(d.ratio(*mul)) + r"\times"
    macros["numBindProj"] = x(d.ratio(bf, pf)) + r"\times"
    macros["numBindUpper"] = x(d.ratio(bf, bu)) + r"\times"
    macros["numBindCombined"] = x(d.ratio(bf, pu)) + r"\times"
    macros["numBindProjBF"] = x(d.ratio(bbf, pbf)) + r"\times"
    macros["numBindBoolBFms"] = ms(d.med(bbf))
    macros["numBindProjBFms"] = ms(d.med(pbf))
    if d.ratio(bf, pf) and d.ratio(bbf, pbf):
        lo, hi = sorted((share(d.ratio(bf, pf)), share(d.ratio(bbf, pbf))))
        macros["numSubShare"] = f"{lo:.0f}\\text{{--}}{hi:.0f}\\%"

    # Table construction (Table 3).
    def table_case(kind, field, rep):
        return f"{kind}_table_build/{field}_{rep}/20"

    rows = []
    for kind, name in (("eq", "EQ"), ("lt", "LT")):
        cells = []
        for field in ("BN254", "BN254bf", "BB4"):
            b, p = table_case(kind, field, "boolean"), table_case(kind, field, "projective")
            cells.append(f"{ms(d.med(b))} & {ms(d.med(p))} (${x(d.ratio(b, p))}\\times$)")
            macros[f"num{name.capitalize()}{ {'BN254': 'BN', 'BN254bf': 'BNbf', 'BB4': 'BB'}[field] }"] = x(d.ratio(b, p))
        rows.append(f"{name} & " + " && ".join(cells) + " \\\\")
    (out / "tab_tables.tex").write_text("\n".join(rows) + "\n")

    # Prover kernels (Table 4).
    variants = ("boolean", "projective", "delayed", "proj_delayed")
    fields = [("BN254 $\\mathbb{F}_r$", "BN254", None),
              ("BN254 $\\mathbb{F}_r$, branch-free", "BN254_bf", None),
              ("BN254 $\\mathbb{F}_r$ + upper-limb", "BN254_upper", None),
              ("Fp128", "Fp128", None),
              ("Fp128, $\\{1,\\infty\\}$", "Fp128", "1inf"),
              ("BB Ext4", "BB4", None), ("BB Ext5", "BB5", None),
              ("KoalaBear Ext5", "KB5", None), ("GF($2^{128}$)", "GF128", None)]
    kernels = ("sumcheck_deg2", "sumcheck_deg2_eq")
    rows = []
    for name, f, var in fields:
        cells = []
        for k in kernels:
            if var:
                cells += ["", ms(d.med(f"{k}_{f}/projective_1inf/20")),
                          "", ms(d.med(f"{k}_{f}/projective_1inf_delayed/20"))]
                continue
            vals = [d.med(f"{k}_{f}/{v}/20") for v in variants]
            best = min((v for v in vals if v is not None), default=None)
            cells += [f"\\textbf{{{ms(v)}}}" if v is not None and v == best else ms(v) for v in vals]
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
    (out / "tab_e2e.tex").write_text("\n".join(rows) + "\n")

    def gain(k, f):
        return d.ratio(f"{k}_{f}/delayed/20", f"{k}_{f}/proj_delayed/20")

    short = {"BN254": "BN", "BN254_bf": "BNbf", "Fp128": "Fp", "BB4": "BBfour",
             "BB5": "BBfive", "KB5": "KB", "GF128": "GF"}
    for f, s in short.items():
        for k, suffix in zip(kernels, ("", "eq")):
            g = gain(k, f)
            macros[f"numProv{s}{suffix}"] = pct(g)
            if g is not None and g < 1:
                # Projective is slower: report how much longer it takes.
                macros[f"numProv{s}{suffix}Loss"] = pct(1 / g)
    for k, suffix in zip(kernels, ("", "eq")):
        macros["numProvUpper" + suffix] = x(d.ratio(f"{k}_BN254/proj_delayed/20",
                                                    f"{k}_BN254_upper/proj_delayed/20")) + r"\times"
        macros["numOneInf" + suffix] = pct(d.ratio(f"{k}_Fp128/proj_delayed/20",
                                                   f"{k}_Fp128/projective_1inf_delayed/20"))

    def pct_range(fs):
        gs = [gain(k, f) for k in kernels for f in fs]
        if any(g is None for g in gs):
            return None
        lo, hi = round((min(gs) - 1) * 100), round((max(gs) - 1) * 100)
        return f"{lo}\\text{{--}}{hi}\\%"

    macros["numProverSpeedup"] = pct_range(("BN254", "Fp128"))
    macros["numProverSpeedupBF"] = pct_range(("BN254_bf",))

    # Field operations (Table 1).  Latency chains have 2^20 operations and
    # throughput batches have 2^10.
    groups = [[("MontyField31", "BabyBear"), ("MontyField31 Ext4", "BabyBear_Ext4"),
               ("MontyField31 Ext5 (bin.)", "BabyBear_Ext5"), ("MontyField31 Ext5 (tri.)", "KoalaBear_Ext5")],
              [("Fp128", "Fp128")], [("BN254 $\\mathbb{F}_r$", "BN254_Fr")], [("GF($2^{128}$)", "GF128_Ghash")]]
    per_op = {}
    lines = []
    for i, group in enumerate(groups):
        if i:
            lines.append("\\addlinespace")
        for name, f in group:
            v = {}
            for kind, n in (("lat", 2 ** 20), ("thr", 2 ** 10)):
                for op in ("add", "sub", "mul"):
                    m = d.med(f"{f}_{kind}_{op}")
                    v[kind, op] = None if m is None else m / n
            per_op[f] = v
            lines.append(f"{name} & " + " & ".join(num(v["lat", o]) for o in ("add", "sub", "mul"))
                         + " && " + " & ".join(num(v["thr", o]) for o in ("add", "sub", "mul")) + " \\\\")
    (out / "tab_fieldops.tex").write_text("\n".join(lines) + "\n")
    bn = per_op["BN254_Fr"]
    if bn["lat", "add"]:
        macros["numSubAdd"] = f"{bn['lat', 'sub'] / bn['lat', 'add']:.1f}"
        macros["numSubMul"] = f"{bn['lat', 'sub'] / bn['lat', 'mul'] * 100:.0f}\\%"
    gaps = [max(v["lat", "add"], v["lat", "sub"]) / min(v["lat", "add"], v["lat", "sub"]) - 1
            for f, v in per_op.items() if f != "BN254_Fr" and v["lat", "add"]]
    if gaps:
        macros["numAddSubGap"] = f"{max(gaps) * 100:.0f}\\%"

    # Uncertainty (appendix table).
    def sc(k, f):
        return f"{k}_{f}/delayed/20", f"{k}_{f}/proj_delayed/20"

    comparisons = [
        ("Binding, projective", bf, pf), ("Binding, projective, branch-free", bbf, pbf),
        ("Binding, upper-limb", bf, bu), ("Binding, both", bf, pu),
        ("Chained multiplication, upper-limb", *mul),
        ("Prover, BN254, degree-2", *sc(kernels[0], "BN254")),
        ("Prover, BN254, degree-2 $\\times$ eq", *sc(kernels[1], "BN254")),
        ("Prover, BN254 branch-free, degree-2", *sc(kernels[0], "BN254_bf")),
        ("Prover, BN254 branch-free, degree-2 $\\times$ eq", *sc(kernels[1], "BN254_bf")),
        ("Prover, Fp128, degree-2", *sc(kernels[0], "Fp128")),
        ("Prover, Fp128, degree-2 $\\times$ eq", *sc(kernels[1], "Fp128")),
        ("EQ table, BN254", table_case("eq", "BN254", "boolean"), table_case("eq", "BN254", "projective")),
        ("EQ table, BN254 branch-free", table_case("eq", "BN254bf", "boolean"), table_case("eq", "BN254bf", "projective")),
        ("LT table, BN254", table_case("lt", "BN254", "boolean"), table_case("lt", "BN254", "projective")),
        ("LT table, BN254 branch-free", table_case("lt", "BN254bf", "boolean"), table_case("lt", "BN254bf", "projective")),
        ("EQ table, BB Ext4", table_case("eq", "BB4", "boolean"), table_case("eq", "BB4", "projective")),
        ("LT table, BB Ext4", table_case("lt", "BB4", "boolean"), table_case("lt", "BB4", "projective")),
    ]
    lines = []
    for name, base, opt in comparisons:
        rs = d.ratios(base, opt)
        if not rs:
            continue
        lo, hi = bootstrap(rs)
        lines.append(f"{name} & {len(rs)} & ${st.median(rs):.3f}$ & $[{lo:.3f},\\,{hi:.3f}]$"
                     f" & $[{min(rs):.3f},\\,{max(rs):.3f}]$ \\\\")
    (out / "tab_uncertainty.tex").write_text("\n".join(lines) + "\n")

    header = ("% Numbers quoted in the text.  Generated from the benchmark data set by\n"
              "% scripts/paper_tables.py in the benchmark repository; do not edit by hand.\n"
              "\\newcommand{\\tbd}{\\text{--}}\n"
              "\\makeatletter\\newcommand{\\tabinput}[1]{\\@@input #1 }\\makeatother\n")
    body = "".join(f"\\newcommand{{\\{k}}}{{{TBD if v is None or TBD in v else v}}}\n"
                   for k, v in macros.items())
    (out / "numbers.tex").write_text(header + body)

    sources = sorted(set(d.src.values()))
    print(f"{len(d.obs)} cases from {len(sources)} run directories")
    for s in sources:
        print("  ", s)


if __name__ == "__main__":
    main()
