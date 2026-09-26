use rftvc_core::{
    Grid, LtrcLogRank, NodeProfile, Profile, SplitCriterion, SurvData, node_profile, profile_on,
};

/// A child profile on the parent's times (exposure and units unused by log-rank).
fn child<'a>(parent: &'a NodeProfile, at_risk: &'a [f64], events: &'a [f64]) -> Profile<'a> {
    Profile {
        at_risk,
        events,
        times: &parent.times,
        exposure: 0.0,
        n_units: 0.0,
    }
}

/// Hand-computed: left = {1 (event), 3 (event)}, right = {2 (event), 4 (censored)}.
/// Per event time (O - E, V): t=1 (1/2, 1/4), t=2 (-1/3, 2/9), t=3 (1/2, 1/4)
/// => chi2 = (2/3)^2 / (13/18) = 8/13.
#[test]
fn logrank_matches_hand_computation() {
    let start = [0.0; 4];
    let stop = [1.0, 3.0, 2.0, 4.0];
    let event = [true, true, true, false];
    let surv = SurvData::new(&start, &stop, &event);
    let parent = node_profile(&surv, &[0, 1, 2, 3]);
    let (l_at, l_ev) = profile_on(&surv, &[0, 1], &parent.event_idx);
    let score = LtrcLogRank.score(&child(&parent, &l_at, &l_ev), &parent.view(4.0), 2.0);
    assert!((score - 8.0 / 13.0).abs() < 1e-12, "score = {score}");
}

/// Index convention: row at risk at k iff start < t_k <= stop.
#[test]
fn grid_index_convention_at_boundaries() {
    let grid = Grid {
        times: std::sync::Arc::new(vec![1.0, 2.0, 3.0]),
    };
    assert_eq!(grid.first_greater(0.5), 0);
    assert_eq!(grid.first_greater(1.0), 1); // start == t_0: not at risk at t_0
    assert_eq!(grid.first_greater(3.0), 3); // stop == t_2: at risk at t_2
    assert_eq!(grid.first_greater(9.0), 3);

    // Delayed entry at exactly an event time, exit exactly at another.
    let surv = SurvData::new(&[0.0, 1.0, 1.5], &[3.0, 2.0, 1.8], &[true, true, false]);
    let p = node_profile(&surv, &[0, 1, 2]);
    // Event times {2, 3}: at t=2 rows 0 and 1 are at risk; row 2 left at 1.8.
    assert_eq!(p.at_risk, vec![2.0, 1.0]);
    assert_eq!(p.events, vec![1.0, 1.0]);
}

#[test]
fn row_with_no_grid_contribution_is_ignored() {
    // Row 1 enters and leaves between event times: contributes to no risk set.
    let surv = SurvData::new(&[0.0, 1.2, 0.0], &[1.0, 1.8, 2.0], &[true, false, true]);
    let p = node_profile(&surv, &[0, 1, 2]);
    assert_eq!(p.at_risk, vec![2.0, 1.0]);
}

#[test]
fn binning_extreme_values_keep_distinct_bins() {
    use rftvc_core::Binned;
    let m = f64::MAX;
    for (a, b) in [(-m, m), (m.next_down(), m), (1.0, 1.0f64.next_up())] {
        let binned = Binned::fit(&[a, b], 2, 1, 256);
        assert_eq!(binned.edges[0].len(), 1);
        let e = binned.edges[0][0];
        assert!(
            e.is_finite() && a <= e && e < b,
            "edge {e} not in [{a}, {b})"
        );
        assert_eq!(binned.column(0), &[0, 1]);
    }
}

/// The precomputed node scorer equals the reference `score` on random
/// delayed-entry data with ties, for every left subset tried.
#[test]
fn node_scorer_matches_reference_score() {
    let mut rng = rftvc_core::Rng::new(7);
    for _ in 0..200 {
        let n = 2 + rng.below(40);
        let start: Vec<f64> = (0..n).map(|_| rng.below(4) as f64).collect();
        let stop: Vec<f64> = start
            .iter()
            .map(|s| s + 1.0 + rng.below(5) as f64)
            .collect();
        let event: Vec<bool> = (0..n).map(|_| rng.below(3) > 0).collect();
        if !event.iter().any(|&e| e) {
            continue;
        }
        let surv = SurvData::new(&start, &stop, &event);
        let rows: Vec<u32> = (0..n as u32).collect();
        let parent = node_profile(&surv, &rows);
        let p = parent.view(n as f64);
        let scorer = LtrcLogRank.node_scorer(parent.view(n as f64));
        let left: Vec<u32> = rows.iter().copied().filter(|_| rng.below(2) == 0).collect();
        let (l_at, l_ev) = profile_on(&surv, &left, &parent.event_idx);
        let l = child(&parent, &l_at, &l_ev);
        let (a, b) = (LtrcLogRank.score(&l, &p, 0.0), scorer.score(&l, 0.0));
        assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0), "{a} vs {b}");
    }
}
