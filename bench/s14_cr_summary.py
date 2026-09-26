"""Summaries of docs/bench/s14-cr/sim.csv for the S14 decision rule (paired over replications)."""

import numpy as np
import polars as pl

from bench.s14_cr_sim import OUT


def paired(df, a, b, cause=None):
    """Mean and SE over replications of ISE(a) - ISE(b) (negative favours a); mean over causes unless ``cause``."""
    d = df if cause is None else df.filter(pl.col("cause") == cause)
    per = d.group_by("rep", "arm").agg(pl.col("ise").mean())
    wide = per.pivot(on="arm", index="rep", values="ise")
    diff = (wide[a] - wide[b]).to_numpy()
    return diff.mean(), diff.std(ddof=1) / np.sqrt(diff.size), diff.size


def table(df, pairs, cause=None):
    rows = []
    for sc in sorted(df["scenario"].unique()):
        d = df.filter(pl.col("scenario") == sc)
        base = d.filter(pl.col("arm") == pairs[0][1]).group_by("rep").agg(pl.col("ise").mean())["ise"].mean()
        for a, b in pairs:
            if a not in d["arm"].unique().to_list():
                continue
            m, se, n = paired(d, a, b, cause)
            rows.append(dict(scenario=sc, arm=a, vs=b, diff=m, se=se, z=m / se if se > 0 else np.nan,
                             rel=m / base, n=n))
    return pl.DataFrame(rows)


def main():
    df = pl.read_csv(OUT / "sim.csv")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_width_chars(140)
    print("C4 (composite): cif vs hazard")
    print(table(df, [("composite/cif", "composite/hazard")]))
    for agg in ("hazard", "cif"):
        print(f"C3 under {agg}: challengers vs composite/{agg}")
        print(table(df, [(f"{c}/{agg}", f"composite/{agg}") for c in ("quadratic", "ishwaran", "logrank_all")]
                    + [("approach_b", f"composite/{agg}")]))
    print("Mean ISE by arm (mean over causes) and fit time")
    print(df.group_by("scenario", "rep", "arm").agg(pl.col("ise").mean(), pl.col("fit_s").first())
          .group_by("scenario", "arm").agg(pl.col("ise").mean(), pl.col("fit_s").mean()).sort("scenario", "arm"))
    c = df.filter(pl.col("scenario") == "C")
    ev = c.filter((pl.col("cause") == 3) & (pl.col("arm") == "split_cause=3")).select("rep", "n_events")
    ok = ev.filter(pl.col("n_events") >= 10)["rep"].to_list()
    print(f"C5: {len(ok)}/{ev.height} replicates with >= 10 cause-3 events; cause-3 events min/median = "
          f"{ev['n_events'].min()}/{ev['n_events'].median()}")
    c_ok = c.filter(pl.col("rep").is_in(ok))
    for m in (3, 5, 10):
        for cause in (3, 1, 2):
            mm, se, n = paired(c_ok, f"split_cause=3/m={m}", "split_cause=3/m=None", cause)
            print(f"  m={m} vs None, cause {cause}: diff={mm:.6f} se={se:.6f} z={mm / se:.2f} n={n}")

if __name__ == "__main__":
    main()
