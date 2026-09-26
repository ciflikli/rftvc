//! Competing risks (S11): criteria, per-cause leaves, Aalen–Johansen prediction, flat v3.

use rftvc_core::{
    Aggregate, Binned, CompositeCauseLogRank, FlatForest, Forest, ForestParams, Groups,
    LtrcLogRank, Profile, Rng, SingleCause, SplitCriterion, SplitParams, SurvData, TreeParams,
    best_split, cause_profile_on, fit_forest, node_profile, profile_on,
};

/// Random delayed-entry rows with half-unit times (many ties) and codes in `0..=n_causes`.
fn random_rows(rng: &mut Rng, n: usize, n_causes: usize) -> (Vec<f64>, Vec<f64>, Vec<u8>) {
    let start: Vec<f64> = (0..n).map(|_| rng.below(6) as f64 * 0.5).collect();
    let stop: Vec<f64> = start
        .iter()
        .map(|s| s + 0.5 + rng.below(8) as f64 * 0.5)
        .collect();
    let codes = (0..n)
        .map(|_| {
            if rng.below(3) > 0 {
                1 + rng.below(n_causes) as u8
            } else {
                0
            }
        })
        .collect();
    (start, stop, codes)
}

fn view<'a>(
    at: &'a [f64],
    ev: &'a [f64],
    cev: &'a [f64],
    nc: usize,
    times: &'a [f64],
) -> Profile<'a> {
    Profile {
        at_risk: at,
        events: ev,
        cause_events: if nc == 1 { ev } else { cev },
        n_causes: nc,
        times,
        exposure: 0.0,
        n_units: 0.0,
    }
}

/// `SingleCause { k }` on J = 3 data equals `LtrcLogRank` on "cause k vs rest"
/// bit-for-bit, each node on its own event times (the J = 3 node has extra
/// all-cause times, which must add exact zeros).
#[test]
fn single_cause_equals_logrank_on_cause_vs_rest() {
    let mut rng = Rng::new(5);
    let mut nonzero = 0;
    for _ in 0..400 {
        let n = 2 + rng.below(40);
        let (start, stop, codes) = random_rows(&mut rng, n, 3);
        let surv = SurvData::with_causes(&start, &stop, &codes, 3);
        let rows: Vec<u32> = (0..n as u32).filter(|_| rng.below(5) > 0).collect();
        let left: Vec<u32> = rows.iter().copied().filter(|_| rng.below(2) == 0).collect();
        let parent = node_profile(&surv, &rows);
        let (l_at, l_ev, l_cev) = cause_profile_on(&surv, &left, &parent.event_idx);
        for k in 0..3 {
            let ev_k: Vec<bool> = codes.iter().map(|&c| c as usize == k + 1).collect();
            let surv_k = SurvData::new(&start, &stop, &ev_k);
            let parent_k = node_profile(&surv_k, &rows);
            let (lk_at, lk_ev) = profile_on(&surv_k, &left, &parent_k.event_idx);
            let want = LtrcLogRank
                .node_scorer(parent_k.view(0.0))
                .score(&view(&lk_at, &lk_ev, &[], 1, &parent_k.times), 0.0);
            let got = SingleCause { cause: k }
                .node_scorer(parent.view(0.0))
                .score(&view(&l_at, &l_ev, &l_cev, 3, &parent.times), 0.0);
            assert_eq!(got.to_bits(), want.to_bits(), "cause {k}: {got} vs {want}");
            // The naive `score` form agrees too.
            let naive = SingleCause { cause: k }.score(
                &view(&l_at, &l_ev, &l_cev, 3, &parent.times),
                &parent.view(0.0),
                0.0,
            );
            assert_eq!(naive.to_bits(), want.to_bits());
            nonzero += (want > 0.0) as usize;
        }
    }
    assert!(nonzero > 100, "{nonzero}");
}

/// Through the splitter: the same best split (feature, bin, score) as the
/// single-event splitter on "cause k vs rest" when every candidate is admissible.
#[test]
fn single_cause_best_split_equals_single_event_forest() {
    let mut rng = Rng::new(8);
    let mut found = 0;
    for _ in 0..200 {
        let n = 4 + rng.below(60);
        let (start, stop, codes) = random_rows(&mut rng, n, 2);
        let x: Vec<f64> = (0..n * 3).map(|_| rng.below(6) as f64).collect();
        let binned = Binned::fit(&x, n, 3, 256);
        let rows: Vec<u32> = (0..n as u32).collect();
        let params = SplitParams {
            min_leaf: 0,
            min_events_leaf: 0,
        };
        let surv = SurvData::with_causes(&start, &stop, &codes, 2);
        for k in 0..2 {
            let ev_k: Vec<bool> = codes.iter().map(|&c| c as usize == k + 1).collect();
            let surv_k = SurvData::new(&start, &stop, &ev_k);
            let crit = SingleCause { cause: k };
            let got = best_split(&binned, &surv, &rows, &rows, &[0, 1, 2], &params, &crit);
            let want = best_split(
                &binned,
                &surv_k,
                &rows,
                &rows,
                &[0, 1, 2],
                &params,
                &LtrcLogRank,
            );
            let key = |s: &rftvc_core::SplitCandidate| (s.feature, s.bin, s.score.to_bits());
            assert_eq!(got.as_ref().map(key), want.as_ref().map(key));
            found += got.is_some() as usize;
        }
    }
    assert!(found > 100, "{found}");
}

/// The composite node scorer equals its naive form, and the naive form equals
/// the sum of per-cause single-event log-rank statistics (each on its own times).
#[test]
fn composite_matches_naive_and_sum_of_causes() {
    let mut rng = Rng::new(13);
    for _ in 0..300 {
        let n = 2 + rng.below(40);
        let nc = 2 + rng.below(3);
        let (start, stop, codes) = random_rows(&mut rng, n, nc);
        let surv = SurvData::with_causes(&start, &stop, &codes, nc);
        let rows: Vec<u32> = (0..n as u32).collect();
        let left: Vec<u32> = rows.iter().copied().filter(|_| rng.below(2) == 0).collect();
        let parent = node_profile(&surv, &rows);
        let (l_at, l_ev, l_cev) = cause_profile_on(&surv, &left, &parent.event_idx);
        let l = view(&l_at, &l_ev, &l_cev, nc, &parent.times);
        let naive = CompositeCauseLogRank.score(&l, &parent.view(0.0), 0.0);
        let fast = CompositeCauseLogRank
            .node_scorer(parent.view(0.0))
            .score(&l, 0.0);
        assert!(
            (naive - fast).abs() <= 1e-12 * naive.max(1.0),
            "{naive} vs {fast}"
        );
        let sum: f64 = (0..nc)
            .map(|k| {
                let ev_k: Vec<bool> = codes.iter().map(|&c| c as usize == k + 1).collect();
                let surv_k = SurvData::new(&start, &stop, &ev_k);
                let p = node_profile(&surv_k, &rows);
                let (a, e) = profile_on(&surv_k, &left, &p.event_idx);
                LtrcLogRank.score(&view(&a, &e, &[], 1, &p.times), &p.view(0.0), 0.0)
            })
            .sum();
        assert!(
            (naive - sum).abs() <= 1e-12 * sum.max(1.0),
            "{naive} vs {sum}"
        );
    }
}

/// A time at which only cause 2 has an event is a grid point.
#[test]
fn grid_has_every_causes_event_times() {
    let surv = SurvData::with_causes(&[0.0; 4], &[1.0, 2.0, 3.0, 4.0], &[1, 2, 0, 2], 2);
    assert_eq!(*surv.grid.times, vec![1.0, 2.0, 4.0]);
}

fn cr_forest(n_trees: usize, max_depth: Option<usize>, seed: u64) -> (Vec<f64>, Forest, SurvData) {
    let mut rng = Rng::new(seed);
    let n = 150;
    let (start, stop, codes) = random_rows(&mut rng, n, 2);
    let x: Vec<f64> = (0..n).map(|_| rng.below(10) as f64).collect();
    let surv = SurvData::with_causes(&start, &stop, &codes, 2);
    let binned = Binned::fit(&x, n, 1, 255);
    let groups = Groups::new(&(0..n as u32).collect::<Vec<_>>(), n);
    let params = ForestParams {
        tree: TreeParams {
            max_depth,
            min_ids_leaf: 5,
            min_events_leaf: 3,
            max_features: 1,
        },
        n_trees,
        n_draw: if n_trees == 1 { n } else { 90 },
        bootstrap: false,
        seed,
    };
    let forest = fit_forest(&binned, &surv, &groups, &params, &CompositeCauseLogRank);
    (x, forest, surv)
}

/// A single root leaf holds the per-cause delayed-entry Nelson–Aalen, and its
/// CIF / survival follow the Aalen–Johansen recursion.
#[test]
fn root_leaf_is_per_cause_nelson_aalen_and_aalen_johansen() {
    let (_, forest, surv) = cr_forest(1, Some(0), 3);
    let tree = &forest.trees[0];
    assert_eq!(tree.n_leaves(), 1);
    let grid = &surv.grid.times;
    let (mut h, mut f, mut s) = ([0.0; 2], [0.0; 2], 1.0);
    let (mut want_h, mut want_f, mut want_s) = (vec![], vec![], vec![]);
    for k in 0..grid.len() {
        let y = (0..surv.n_rows())
            .filter(|&r| (surv.a[r] as usize) <= k && k < surv.b[r] as usize)
            .count() as f64;
        let mut dsum = 0.0;
        for j in 0..2 {
            let d = (0..surv.n_rows())
                .filter(|&r| surv.b[r] as usize == k + 1 && surv.event[r] as usize == j + 1)
                .count() as f64;
            h[j] += d / y;
            f[j] += s * d / y;
            dsum += d / y;
        }
        s *= 1.0 - dsum;
        want_h.extend_from_slice(&h);
        want_f.push(f);
        want_s.push(s);
    }
    for (a, b) in tree.leaf_cumhaz(0).iter().zip(&want_h) {
        assert!((a - b).abs() < 1e-12);
    }
    assert_eq!(tree.leaf_cumhaz(0).len(), grid.len() * 2);
    let (cif, sv, clamped) = forest.predict_cif(&[0.0], 1, grid);
    let m = grid.len();
    assert_eq!(clamped, 0);
    for k in 0..m {
        assert!((sv[k] - want_s[k]).abs() < 1e-12);
        for j in 0..2 {
            assert!((cif[j * m + k] - want_f[k][j]).abs() < 1e-12);
        }
    }
}

/// On a forest: F_1 + F_2 + S = 1, F non-decreasing, S non-increasing, and
/// unsorted / repeated / off-grid requested times give the per-time values.
#[test]
fn aalen_johansen_invariants_and_time_order() {
    let (x, forest, surv) = cr_forest(25, None, 11);
    let grid = surv.grid.times.to_vec();
    let m = grid.len();
    let n = 40;
    let (cif, sv, clamped) = forest.predict_cif(&x[..n], 1, &grid);
    assert_eq!(clamped, 0);
    for r in 0..n {
        let (c, s) = (&cif[r * 2 * m..(r + 1) * 2 * m], &sv[r * m..(r + 1) * m]);
        for k in 0..m {
            assert!((c[k] + c[m + k] + s[k] - 1.0).abs() < 1e-12);
            if k > 0 {
                assert!(c[k] >= c[k - 1] && c[m + k] >= c[m + k - 1] && s[k] <= s[k - 1]);
            }
        }
    }
    let times = [grid[m / 2] + 0.25, -1.0, grid[m - 1], grid[3], grid[3], 1e9];
    let (c2, s2, _) = forest.predict_cif(&x[..n], 1, &times);
    for r in 0..n {
        for (ti, &t) in times.iter().enumerate() {
            let pos = grid.partition_point(|&g| g <= t);
            let (want_s, want_c) = if pos == 0 {
                (1.0, [0.0, 0.0])
            } else {
                let k = pos - 1;
                (sv[r * m + k], [cif[r * 2 * m + k], cif[r * 2 * m + m + k]])
            };
            assert_eq!(s2[r * times.len() + ti], want_s);
            for j in 0..2 {
                assert_eq!(c2[r * 2 * times.len() + j * times.len() + ti], want_c[j]);
            }
        }
    }
}

/// With one cause, `predict_cause_cumhaz` equals `predict_cumhaz(Hazard)` bit-for-bit.
#[test]
fn one_cause_cumhaz_matches_single_event_prediction() {
    let mut rng = Rng::new(21);
    let n = 120;
    let (start, stop, codes) = random_rows(&mut rng, n, 1);
    let x: Vec<f64> = (0..n).map(|_| rng.below(10) as f64).collect();
    let surv = SurvData::with_causes(&start, &stop, &codes, 1);
    let binned = Binned::fit(&x, n, 1, 255);
    let groups = Groups::new(&(0..n as u32).collect::<Vec<_>>(), n);
    let params = ForestParams {
        tree: TreeParams {
            max_depth: None,
            min_ids_leaf: 5,
            min_events_leaf: 3,
            max_features: 1,
        },
        n_trees: 15,
        n_draw: 70,
        bootstrap: false,
        seed: 4,
    };
    let forest = fit_forest(&binned, &surv, &groups, &params, &LtrcLogRank);
    let t = surv.grid.times.to_vec();
    let a = forest.predict_cause_cumhaz(&x, 1, &t);
    let b = forest.predict_cumhaz(&x, 1, &t, Aggregate::Hazard);
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
}

/// A leaf whose whole risk set fails at its last time gives S = 0 there.
#[test]
fn whole_risk_set_failing_gives_zero_survival() {
    let surv = SurvData::with_causes(&[0.0, 0.0], &[1.0, 2.0], &[1, 2], 2);
    let binned = Binned::fit(&[0.0, 0.0], 2, 1, 255);
    let groups = Groups::new(&[0, 1], 2);
    let params = ForestParams {
        tree: TreeParams {
            max_depth: Some(0),
            min_ids_leaf: 1,
            min_events_leaf: 1,
            max_features: 1,
        },
        n_trees: 1,
        n_draw: 2,
        bootstrap: false,
        seed: 0,
    };
    let forest = fit_forest(&binned, &surv, &groups, &params, &CompositeCauseLogRank);
    let (cif, sv, clamped) = forest.predict_cif(&[0.0], 1, &[1.0, 2.0]);
    assert!(clamped <= 1);
    assert_eq!(sv, vec![0.5, 0.0]);
    assert_eq!(cif, vec![0.5, 0.5, 0.0, 0.5]);
}

fn cr_flat() -> FlatForest {
    FlatForest::from_forest(&cr_forest(5, None, 17).1)
}

#[test]
fn flat_v3_roundtrip_preserves_cif() {
    let (x, forest, surv) = cr_forest(5, None, 17);
    let flat = FlatForest::from_forest(&forest);
    assert_eq!((flat.format_version, flat.n_causes), (3, 2));
    let back = flat.to_forest().unwrap();
    let t = surv.grid.times.to_vec();
    assert_eq!(forest.predict_cif(&x, 1, &t), back.predict_cif(&x, 1, &t));
    assert_eq!(FlatForest::from_forest(&back), flat);
}

#[test]
fn flat_v2_loads_as_one_cause_and_bad_v3_states_error() {
    // A v2 state is a single-event state without `n_causes` (the loader passes 1).
    let single = {
        let mut rng = Rng::new(2);
        let (start, stop, codes) = random_rows(&mut rng, 80, 1);
        let x: Vec<f64> = (0..80).map(|_| rng.below(10) as f64).collect();
        let surv = SurvData::with_causes(&start, &stop, &codes, 1);
        let binned = Binned::fit(&x, 80, 1, 255);
        let groups = Groups::new(&(0..80u32).collect::<Vec<_>>(), 80);
        let params = ForestParams {
            tree: TreeParams {
                max_depth: None,
                min_ids_leaf: 5,
                min_events_leaf: 3,
                max_features: 1,
            },
            n_trees: 3,
            n_draw: 50,
            bootstrap: false,
            seed: 1,
        };
        fit_forest(&binned, &surv, &groups, &params, &LtrcLogRank)
    };
    let mut v2 = FlatForest::from_forest(&single);
    v2.format_version = 2;
    assert_eq!(v2.to_forest().unwrap().n_causes, 1);
    v2.n_causes = 2;
    assert!(v2.to_forest().is_err(), "v2 with two causes");

    let flat = cr_flat();
    let mut bad = flat.clone();
    bad.n_causes = 0;
    assert!(bad.to_forest().is_err(), "n_causes = 0");
    let mut bad = flat.clone();
    bad.n_causes = 3;
    assert!(bad.to_forest().is_err(), "wrong stride");
    let mut bad = flat.clone();
    bad.cumhaz.pop();
    assert!(bad.to_forest().is_err(), "short cumhaz");
    let mut bad = flat.clone();
    bad.format_version = 4;
    assert!(bad.to_forest().is_err(), "future version");
    // Cause 2 decreasing inside a leaf while cause 1 stays valid.
    let leaf = (0..flat.event_offsets.len() - 1)
        .find(|&l| {
            let (a, b) = (
                flat.event_offsets[l] as usize,
                flat.event_offsets[l + 1] as usize,
            );
            b - a >= 2 && flat.cumhaz[2 * (a + 1) + 1] > 0.0
        })
        .unwrap();
    let e0 = flat.event_offsets[leaf] as usize;
    let mut bad = flat.clone();
    bad.cumhaz[2 * e0 + 1] = bad.cumhaz[2 * (e0 + 1) + 1] + 1.0;
    assert!(bad.to_forest().is_err(), "decreasing cause-2 column");
}
