use rftvc_core::{
    Aggregate, Binned, FlatForest, ForestParams, Groups, SurvData, TreeParams, draw_ids, fit_forest,
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
        let forest = fit_forest(&binned, &surv, &groups, &params(bootstrap));
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
    let flat = FlatForest::from_forest(&fit_forest(&binned, &surv, &groups, &params(false)));
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
