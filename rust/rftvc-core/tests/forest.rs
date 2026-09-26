use rftvc_core::{
    Aggregate, Binned, FlatForest, ForestParams, Groups, LtrcLogRank, SurvData, TreeParams,
    draw_ids, fit_forest,
};

fn toy(n: usize) -> (Vec<f64>, SurvData) {
    let x: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
    let stop: Vec<f64> = (0..n)
        .map(|i| 1.0 + ((i * 37) % 23) as f64 + (i % 7) as f64)
        .collect();
    let event: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
    (x, SurvData::new(&vec![0.0; n], &stop, &event))
}

fn params(bootstrap: bool) -> ForestParams {
    ForestParams {
        tree: TreeParams {
            max_depth: None,
            min_ids_leaf: 5,
            min_events_leaf: 2,
            max_features: 1,
        },
        n_trees: 20,
        n_draw: 60,
        bootstrap,
        seed: 42,
    }
}

#[test]
fn subsample_draws_distinct_ids() {
    let ids = draw_ids(7, 100, 63, false);
    let mut u = ids.clone();
    u.dedup();
    assert_eq!(u.len(), 63);
    assert!(ids.iter().all(|&g| g < 100));
}

#[test]
fn bootstrap_draws_with_replacement() {
    let ids = draw_ids(7, 50, 50, true);
    let mut u = ids.clone();
    u.dedup();
    assert_eq!(ids.len(), 50);
    assert!(u.len() < 50);
}

#[test]
fn flat_roundtrip_preserves_predictions() {
    let n = 100;
    let (x, surv) = toy(n);
    let binned = Binned::fit(&x, n, 1, 255);
    let groups = Groups::new(&(0..n as u32).collect::<Vec<_>>(), n);
    let times: Vec<f64> = (0..40).map(|t| t as f64).collect();
    for bootstrap in [false, true] {
        let forest = fit_forest(&binned, &surv, &groups, &params(bootstrap), &LtrcLogRank);
        let flat = FlatForest::from_forest(&forest);
        let back = flat.to_forest().unwrap();
        assert_eq!(FlatForest::from_forest(&back), flat);
        for agg in [Aggregate::Hazard, Aggregate::Survival] {
            assert_eq!(
                forest.predict_cumhaz(&x, 1, &times, agg),
                back.predict_cumhaz(&x, 1, &times, agg)
            );
        }
        assert_eq!(forest.in_bag_ids(3), back.in_bag_ids(3));
    }
}

#[test]
fn corrupt_flat_state_errors() {
    let n = 60;
    let (x, surv) = toy(n);
    let binned = Binned::fit(&x, n, 1, 255);
    let groups = Groups::new(&(0..n as u32).collect::<Vec<_>>(), n);
    let flat = FlatForest::from_forest(&fit_forest(
        &binned,
        &surv,
        &groups,
        &params(false),
        &LtrcLogRank,
    ));
    let mut bad = flat.clone();
    bad.node_left[0] = u32::MAX;
    assert!(bad.to_forest().is_err());
    let mut bad = flat.clone();
    bad.event_idx.push(0);
    assert!(bad.to_forest().is_err());
    let mut bad = flat;
    bad.tree_seeds.pop();
    assert!(bad.to_forest().is_err());
}

fn valid_flat() -> FlatForest {
    let n = 60;
    let (x, surv) = toy(n);
    let binned = Binned::fit(&x, n, 1, 255);
    let groups = Groups::new(&(0..n as u32).collect::<Vec<_>>(), n);
    FlatForest::from_forest(&fit_forest(
        &binned,
        &surv,
        &groups,
        &params(false),
        &LtrcLogRank,
    ))
}

#[test]
fn structurally_consistent_but_invalid_states_error() {
    // An empty tree: offsets agree with each other but there is no root.
    let empty = FlatForest {
        grid: vec![1.0],
        tree_seeds: vec![0],
        n_features: 1,
        n_groups: 1,
        n_draw: 1,
        node_offsets: vec![0, 0],
        leaf_offsets: vec![0, 0],
        event_offsets: vec![0],
        ..Default::default()
    };
    assert!(empty.to_forest().is_err());
    // No trees at all.
    let none = FlatForest {
        node_offsets: vec![0],
        leaf_offsets: vec![0],
        event_offsets: vec![0],
        ..Default::default()
    };
    assert!(none.to_forest().is_err());

    let flat = valid_flat();
    let split = flat.node_feature.iter().position(|&f| f >= 0).unwrap();
    let mut bad = flat.clone();
    bad.node_threshold[split] = f64::NAN;
    assert!(bad.to_forest().is_err(), "NaN threshold");

    let leaf = (0..flat.event_offsets.len() - 1)
        .find(|&l| flat.event_offsets[l + 1] - flat.event_offsets[l] >= 2)
        .unwrap();
    let e0 = flat.event_offsets[leaf] as usize;
    let mut bad = flat.clone();
    bad.event_idx.swap(e0, e0 + 1);
    assert!(bad.to_forest().is_err(), "unsorted event times");
    let mut bad = flat.clone();
    bad.d[e0] = bad.y[e0] + 1.0;
    assert!(bad.to_forest().is_err(), "d > y");
    let mut bad = flat;
    bad.y[e0] = 0.0;
    assert!(bad.to_forest().is_err(), "y = 0");
}

#[test]
fn survival_aggregation_is_finite_for_large_hazards() {
    // Two single-leaf trees whose cumulative hazard reaches 1000 (1000 events, d = y = 1).
    let k = 1000u32;
    let flat = FlatForest {
        grid: (0..k).map(|t| t as f64 + 1.0).collect(),
        tree_seeds: vec![1, 2],
        n_features: 1,
        n_groups: 1,
        n_draw: 1,
        node_offsets: vec![0, 1, 2],
        node_feature: vec![-1, -1],
        node_threshold: vec![0.0, 0.0],
        node_left: vec![0, 0],
        node_right: vec![0, 0],
        leaf_offsets: vec![0, 1, 2],
        event_offsets: vec![0, k as u64, 2 * k as u64],
        event_idx: (0..k).chain(0..k).collect(),
        d: vec![1.0; 2 * k as usize],
        y: vec![1.0; 2 * k as usize],
        ..Default::default()
    };
    let forest = flat.to_forest().unwrap();
    let t = [k as f64, 5.0];
    let hz = forest.predict_cumhaz(&[0.0], 1, &t, Aggregate::Hazard);
    let sv = forest.predict_cumhaz(&[0.0], 1, &t, Aggregate::Survival);
    assert_eq!(hz, vec![1000.0, 5.0]);
    for (s, h) in sv.iter().zip(&hz) {
        assert!(
            s.is_finite() && (s - h).abs() < 1e-9,
            "survival agg {s} vs {h}"
        );
    }
}
