use rftvc_core::{
    GroupedLik, KmGini, PoissonExposure, Profile, SplitCriterion, SurvData, criterion, exposure_of,
    node_profile, profile_on,
};

/// Rows (0, stop] with stop = [1, 3, 2, 4], events [1, 1, 1, 0]; left = {0, 1},
/// one unit per row. Parent event times 1, 2, 3: y = [4, 3, 2], d = [1, 1, 1];
/// left y = [2, 1, 1], d = [1, 0, 1]; right y = [2, 2, 1], d = [0, 1, 0].
fn score(c: &dyn SplitCriterion) -> f64 {
    let surv = SurvData::new(&[0.0; 4], &[1.0, 3.0, 2.0, 4.0], &[true, true, true, false]);
    let parent = node_profile(&surv, &[0, 1, 2, 3]);
    let (l_at, l_ev) = profile_on(&surv, &[0, 1], &parent.event_idx);
    let left = Profile {
        at_risk: &l_at,
        events: &l_ev,
        times: &parent.times,
        exposure: exposure_of(&surv, &[0, 1]),
        n_units: 2.0,
    };
    c.score(&left, &parent.view(4.0), 2.0)
}

/// ll(y, d) = d ln(d/y) + (y-d) ln(1 - d/y). Children: ll(2,1) = 2 ln(1/2)
/// (left t=1, right t=2), all other child terms 0. Parent:
/// ll(4,1) + ll(3,1) + ll(2,1) = ln(1/4) + 3 ln(3/4) + ln(1/3) + 2 ln(2/3) + 2 ln(1/2).
/// Gain = 4 ln 2.
#[test]
fn grouped_lik_matches_hand_computation() {
    let got = score(&GroupedLik);
    assert!((got - 4.0 * 2f64.ln()).abs() < 1e-12, "{got}");
}

/// D/E: left 2/4, right 1/6, parent 3/10.
/// Gain = 2 ln(1/2) + ln(1/6) - 3 ln(3/10).
#[test]
fn poisson_matches_hand_computation() {
    let want = 2.0 * 0.5f64.ln() + (1.0f64 / 6.0).ln() - 3.0 * 0.3f64.ln();
    let got = score(&PoissonExposure);
    assert!((got - want).abs() < 1e-12, "{got} vs {want}");
}

/// tau = 3 includes the event at 3 (right-continuous): S_parent = 3/4 · 2/3 · 1/2
/// = 1/4, S_left = 1/2 · 1 · 0 = 0, S_right = 1 · 1/2 · 1 = 1/2.
/// Gain = 4·(1/4)(3/4) - 0 - 2·(1/2)(1/2) = 1/4.
/// tau = 2.5 stops before t = 3: all S = 1/2, gain = 1 - 1/2 - 1/2 = 0.
/// tau = 0.5 precedes every event: S = 1 everywhere, gain 0.
#[test]
fn km_gini_matches_hand_computation() {
    for (tau, want) in [(3.0, 0.25), (2.5, 0.0), (0.5, 0.0), (9.0, 0.25)] {
        let got = score(&KmGini { horizon: tau });
        assert!((got - want).abs() < 1e-12, "tau {tau}: {got} vs {want}");
    }
}

#[test]
fn criterion_by_name_validates_horizon() {
    assert!(criterion("logrank", None).is_ok());
    assert!(criterion("grouped_lik", None).is_ok());
    assert!(criterion("poisson", None).is_ok());
    assert!(criterion("km_gini", Some(2.0)).is_ok());
    for (name, h) in [
        ("km_gini", None),
        ("km_gini", Some(0.0)),
        ("km_gini", Some(f64::NAN)),
        ("km_gini", Some(f64::INFINITY)),
        ("logrank", Some(1.0)),
        ("poisson", Some(1.0)),
        ("gini", None),
    ] {
        assert!(criterion(name, h).is_err(), "{name} {h:?}");
    }
}
