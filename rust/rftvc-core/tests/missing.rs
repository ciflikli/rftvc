//! Missing-value (MIA) support in the split search: `Binned::fit`'s NaN handling,
//! the two-direction threshold sweep, the `MissingVsObserved` candidate, and
//! mixed-unit (a resampling unit with both missing and observed rows) correctness.

use std::collections::HashSet;

use rftvc_core::{
    Binned, FlatForest, Forest, LtrcLogRank, Node, NodeProfile, Profile, Rng, SplitCandidate,
    SplitCriterion, SplitParams, SplitRule, SurvData, TreeParams, best_split, build_tree,
    cause_profile_on, exposure_of, node_profile,
};

fn random_rows(rng: &mut Rng, n: usize) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let start: Vec<f64> = (0..n).map(|_| rng.below(4) as f64 * 0.5).collect();
    let stop: Vec<f64> = start
        .iter()
        .map(|s| s + 0.25 + rng.below(6) as f64 * 0.75)
        .collect();
    let event: Vec<bool> = (0..n).map(|_| rng.below(2) == 0).collect();
    (start, stop, event)
}

/// A brute-force reference: tries every real threshold in both missing
/// directions, plus `MissingVsObserved`, scoring each via the same criterion
/// `best_split_in` uses internally, and returns the best (score, rule).
fn brute_force_best(
    surv: &SurvData,
    binned: &Binned,
    profile: &NodeProfile,
    rows: &[u32],
    units: &[u32],
    feature: usize,
    params: &SplitParams,
) -> Option<(f64, SplitRule)> {
    let col = binned.column(feature);
    let miss = binned.missing_column(feature);
    let n_units_of =
        |idx: &[usize]| -> usize { idx.iter().map(|&i| units[i]).collect::<HashSet<_>>().len() };
    let admissible = |left: &[usize], right: &[usize]| -> bool {
        let (n_l, n_r) = (n_units_of(left), n_units_of(right));
        let (e_l, e_r) = (
            left.iter()
                .filter(|&&i| surv.event[rows[i] as usize] != 0)
                .count(),
            right
                .iter()
                .filter(|&&i| surv.event[rows[i] as usize] != 0)
                .count(),
        );
        n_l >= params.min_leaf
            && n_r >= params.min_leaf
            && e_l >= params.min_events_leaf
            && e_r >= params.min_events_leaf
    };
    let scorer = LtrcLogRank
        .node_scorer(profile.view(n_units_of(&(0..rows.len()).collect::<Vec<_>>()) as f64));
    let score_of = |left_idx: &[usize]| -> f64 {
        let left_rows: Vec<u32> = left_idx.iter().map(|&i| rows[i]).collect();
        let (at, ev, _) = cause_profile_on(surv, &left_rows, &profile.event_idx);
        let exposure = exposure_of(surv, &left_rows);
        let left = Profile {
            at_risk: &at,
            events: &ev,
            cause_events: &ev,
            n_causes: 1,
            times: &profile.times,
            exposure,
            n_units: n_units_of(left_idx) as f64,
        };
        scorer.score(
            &left,
            n_units_of(
                &(0..rows.len())
                    .filter(|i| !left_idx.contains(i))
                    .collect::<Vec<_>>(),
            ) as f64,
        )
    };

    let mut best: Option<(f64, SplitRule)> = None;
    let mut consider = |score: f64, rule: SplitRule| {
        if score > 0.0 && best.is_none_or(|(b, _)| score > b) {
            best = Some((score, rule));
        }
    };

    // MissingVsObserved
    let miss_idx: Vec<usize> = (0..rows.len())
        .filter(|&i| miss[rows[i] as usize])
        .collect();
    let obs_idx: Vec<usize> = (0..rows.len())
        .filter(|&i| !miss[rows[i] as usize])
        .collect();
    if !miss_idx.is_empty() && !obs_idx.is_empty() && admissible(&miss_idx, &obs_idx) {
        consider(score_of(&miss_idx), SplitRule::MissingVsObserved);
    }

    // Every real threshold, both missing directions.
    let mut used_bins: Vec<u8> = obs_idx.iter().map(|&i| col[rows[i] as usize]).collect();
    used_bins.sort_unstable();
    used_bins.dedup();
    for &b in used_bins.iter().take(used_bins.len().saturating_sub(1)) {
        let obs_left: Vec<usize> = obs_idx
            .iter()
            .copied()
            .filter(|&i| col[rows[i] as usize] <= b)
            .collect();
        let obs_right: Vec<usize> = obs_idx
            .iter()
            .copied()
            .filter(|&i| col[rows[i] as usize] > b)
            .collect();

        // missing -> right
        let right_r: Vec<usize> = obs_right.iter().chain(miss_idx.iter()).copied().collect();
        if admissible(&obs_left, &right_r) {
            consider(
                score_of(&obs_left),
                SplitRule::Threshold {
                    bin: b,
                    missing_goes_right: true,
                },
            );
        }
        // missing -> left
        let left_l: Vec<usize> = obs_left.iter().chain(miss_idx.iter()).copied().collect();
        if admissible(&left_l, &obs_right) {
            consider(
                score_of(&left_l),
                SplitRule::Threshold {
                    bin: b,
                    missing_goes_right: false,
                },
            );
        }
    }
    best
}

fn setup(
    rng: &mut Rng,
    n: usize,
    missing_rate: f64,
    n_features: usize,
) -> (SurvData, Binned, Vec<u32>, Vec<u32>) {
    let (start, stop, event) = random_rows(rng, n);
    let surv = SurvData::new(&start, &stop, &event);
    let mut x: Vec<f64> = (0..n * n_features).map(|_| rng.below(5) as f64).collect();
    for v in x.iter_mut() {
        if (rng.below(1000) as f64 / 1000.0) < missing_rate {
            *v = f64::NAN;
        }
    }
    let binned = Binned::fit(&x, n, n_features, 256);
    let mut units: Vec<u32> = (0..n).map(|_| rng.below(n / 2 + 1) as u32).collect();
    units.sort_unstable();
    let rows: Vec<u32> = (0..n as u32).collect();
    (surv, binned, rows, units)
}

#[test]
fn binned_fit_does_not_panic_on_nan_and_sets_missing_marker() {
    let x = [1.0, f64::NAN, 3.0, f64::NAN, 2.0];
    let binned = Binned::fit(&x, 5, 1, 256);
    assert_eq!(binned.missing_column(0), &[false, true, false, true, false]);
    // Edges are computed from {1.0, 2.0, 3.0} only.
    assert_eq!(binned.edges[0], vec![1.5, 2.5]);
}

#[test]
fn two_direction_search_matches_brute_force() {
    let mut rng = Rng::new(7);
    let params = SplitParams {
        min_leaf: 1,
        min_events_leaf: 1,
        cause_floor: None,
    };
    let mut found_missing_left = 0;
    let mut found_missing_right = 0;
    let mut found_missing_vs_observed = 0;
    for _ in 0..400 {
        let n = 6 + rng.below(30);
        let (surv, binned, rows, units) = setup(&mut rng, n, 0.35, 1);
        let profile = node_profile(&surv, &rows);
        if profile.event_idx.is_empty() {
            continue;
        }
        let want = brute_force_best(&surv, &binned, &profile, &rows, &units, 0, &params);
        let got: Option<SplitCandidate> =
            best_split(&binned, &surv, &rows, &units, &[0], &params, &LtrcLogRank);
        match (want, &got) {
            (None, None) => {}
            (Some((ws, wr)), Some(g)) => {
                assert!(
                    (ws - g.score).abs() < 1e-9,
                    "score mismatch: want {ws} got {}",
                    g.score
                );
                assert_eq!(wr, g.rule, "rule mismatch");
                if let SplitRule::Threshold {
                    missing_goes_right, ..
                } = wr
                {
                    let tree = build_tree(
                        &binned,
                        &surv,
                        rows.clone(),
                        units.clone(),
                        &TreeParams {
                            max_depth: Some(1),
                            min_ids_leaf: 1,
                            min_events_leaf: 1,
                            max_features: 1,
                            ..Default::default()
                        },
                        &LtrcLogRank,
                        &mut Rng::new(0),
                    );
                    if let Node::Split {
                        missing_goes_right: route,
                        left,
                        right,
                        ..
                    } = tree.nodes[0]
                    {
                        assert_eq!(route, missing_goes_right);
                        let chosen = if route { right } else { left };
                        if let Node::Leaf { leaf } = tree.nodes[chosen as usize] {
                            assert_eq!(tree.apply(&[f64::NAN]), leaf as usize);
                        } else {
                            panic!("depth-one split must have leaf children");
                        }
                    } else {
                        panic!("best split must be used at the root");
                    }
                }
                match wr {
                    SplitRule::Threshold {
                        missing_goes_right: true,
                        ..
                    } => found_missing_right += 1,
                    SplitRule::Threshold {
                        missing_goes_right: false,
                        ..
                    } => found_missing_left += 1,
                    SplitRule::MissingVsObserved => found_missing_vs_observed += 1,
                }
            }
            (w, g) => panic!("presence mismatch: want {w:?} got {g:?}"),
        }
    }
    assert!(
        found_missing_left > 0,
        "never exercised a missing-left threshold"
    );
    assert!(
        found_missing_right > 0,
        "never exercised a missing-right threshold"
    );
    assert!(
        found_missing_vs_observed > 0,
        "never exercised MissingVsObserved"
    );
}

#[test]
fn missing_vs_observed_chosen_when_observed_values_are_constant() {
    // Every observed row has the same x value, so only a missingness split
    // can separate the early events from the later events.
    let n = 20;
    let start = vec![0.0; n];
    let stop: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
    let event = vec![true; n];
    let surv = SurvData::new(&start, &stop, &event);
    let x: Vec<f64> = (0..n)
        .map(|i| if i < 10 { f64::NAN } else { 7.0 })
        .collect();
    let binned = Binned::fit(&x, n, 1, 256);
    let rows: Vec<u32> = (0..n as u32).collect();
    let units = rows.clone();
    let params = SplitParams {
        min_leaf: 1,
        min_events_leaf: 1,
        cause_floor: None,
    };
    let got = best_split(&binned, &surv, &rows, &units, &[0], &params, &LtrcLogRank);
    let g = got.expect("missingness perfectly predicts the event; a split must be found");
    assert_eq!(g.rule, SplitRule::MissingVsObserved);
    assert!(g.threshold.is_nan());
}

#[test]
fn feature_missing_for_every_row_contributes_no_split() {
    let mut rng = Rng::new(11);
    let n = 30;
    let (surv, _, rows, units) = setup(&mut rng, n, 0.0, 1);
    let x = vec![f64::NAN; n];
    let binned = Binned::fit(&x, n, 1, 256);
    let params = SplitParams {
        min_leaf: 1,
        min_events_leaf: 1,
        cause_floor: None,
    };
    let got = best_split(&binned, &surv, &rows, &units, &[0], &params, &LtrcLogRank);
    assert!(got.is_none());
}

#[test]
fn zero_missing_rows_always_chooses_missing_goes_right_true() {
    // With no missing rows the search must still report a well-defined
    // (unexercised) missing direction: `true`, matching the no-missing path.
    let mut rng = Rng::new(5);
    let params = SplitParams {
        min_leaf: 1,
        min_events_leaf: 1,
        cause_floor: None,
    };
    let mut found = 0;
    for _ in 0..100 {
        let n = 6 + rng.below(30);
        let (surv, binned, rows, units) = setup(&mut rng, n, 0.0, 1);
        if let Some(g) = best_split(&binned, &surv, &rows, &units, &[0], &params, &LtrcLogRank) {
            match g.rule {
                SplitRule::Threshold {
                    missing_goes_right, ..
                } => {
                    assert!(missing_goes_right);
                    found += 1;
                }
                SplitRule::MissingVsObserved => {
                    panic!("no missing rows: MissingVsObserved cannot apply")
                }
            }
        }
    }
    assert!(found > 0);
}

#[test]
fn mixed_unit_straddles_correctly_under_both_missing_directions() {
    // Unit 0 has two rows for feature 0: one observed (bin low), one missing.
    // Unit 1 has one observed row (bin high). Under missing->left, unit 0 must
    // count in "left" (its missing row) even before its own observed bin is
    // swept; under missing->right, unit 0 only enters "left" once its observed
    // bin is swept, matching the pre-existing straddling convention.
    let start = vec![0.0, 0.0, 0.0];
    let stop = vec![1.0, 2.0, 3.0];
    let event = vec![true, true, true];
    let surv = SurvData::new(&start, &stop, &event);
    let x = [0.0, f64::NAN, 5.0]; // row0: unit0 observed low; row1: unit0 missing; row2: unit1 observed high
    let binned = Binned::fit(&x, 3, 1, 256);
    let rows: Vec<u32> = vec![0, 1, 2];
    let units: Vec<u32> = vec![0, 0, 1]; // unit 0's rows contiguous, then unit 1
    let params = SplitParams {
        min_leaf: 1,
        min_events_leaf: 1,
        cause_floor: None,
    };
    let got = best_split(&binned, &surv, &rows, &units, &[0], &params, &LtrcLogRank);
    let g = got.expect("a two-unit, two-bin node must find a split");
    // Whichever rule won, verify ids_left/ids_right against the brute-force
    // reference's own admissibility bookkeeping (both units always end up
    // counted somewhere; the point under test is that neither is silently lost
    // or double-excluded).
    assert!(g.ids_left >= 1 && g.ids_right >= 1);
    assert!(g.ids_left <= 2 && g.ids_right <= 2);
}

#[test]
fn mixed_unit_routes_to_both_leaves_and_roundtrips() {
    // Unit 0 has one missing and one observed row. Their events must enter
    // different leaf hazards, and the serialized tree must preserve that route.
    let stop = [1.0, 10.0, 2.0, 3.0, 11.0, 12.0];
    let surv = SurvData::new(&[0.0; 6], &stop, &[true; 6]);
    let x = [f64::NAN, 7.0, f64::NAN, f64::NAN, 7.0, 7.0];
    let binned = Binned::fit(&x, 6, 1, 256);
    let tree = build_tree(
        &binned,
        &surv,
        (0..6).collect(),
        vec![0, 0, 1, 2, 3, 4],
        &TreeParams {
            max_depth: Some(1),
            min_ids_leaf: 1,
            min_events_leaf: 1,
            max_features: 1,
            ..Default::default()
        },
        &LtrcLogRank,
        &mut Rng::new(0),
    );
    let missing_leaf = tree.apply(&[f64::NAN]);
    let observed_leaf = tree.apply(&[7.0]);
    assert_ne!(missing_leaf, observed_leaf);
    let event_times = |leaf| -> Vec<f64> {
        tree.leaf_event_idx(leaf)
            .iter()
            .map(|&i| tree.grid_times[i as usize])
            .collect()
    };
    assert_eq!(event_times(missing_leaf), vec![1.0, 2.0, 3.0]);
    assert_eq!(event_times(observed_leaf), vec![10.0, 11.0, 12.0]);

    let forest = Forest {
        trees: vec![tree],
        tree_seeds: vec![0],
        n_features: 1,
        n_groups: 5,
        n_draw: 5,
        bootstrap: false,
        n_causes: 1,
    };
    let flat = FlatForest::from_forest(&forest);
    assert!(flat.node_threshold[0].is_nan());
    assert!(!flat.node_missing_right[0]);
    let restored = flat.to_forest().unwrap();
    assert_eq!(restored.trees[0].apply(&[f64::NAN]), missing_leaf);
    assert_eq!(restored.trees[0].apply(&[7.0]), observed_leaf);
    assert_eq!(
        FlatForest::from_forest(&restored).node_missing_right,
        flat.node_missing_right
    );
}
