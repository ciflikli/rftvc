use crate::criterion::{Profile, SplitCriterion};

use crate::data::{Binned, SurvData};

/// A node's risk-set profile on its own event times.
///
/// Only times with at least one event in the node enter the log-rank sum, so
/// the node works on `event_idx` (global grid indices, sorted) rather than the full grid.
#[derive(Clone, Debug)]
pub struct NodeProfile {
    pub event_idx: Vec<u32>,
    pub at_risk: Vec<f64>,
    pub events: Vec<f64>,
    /// Events per cause, cause-major `[j * K + k]`; empty when `n_causes == 1`
    /// (the view then uses `events`).
    pub cause_events: Vec<f64>,
    pub n_causes: usize,
    /// Grid times of `event_idx`.
    pub times: Vec<f64>,
    /// Person-time of the node's rows.
    pub exposure: f64,
    pub n_events: usize,
    /// Events of each cause (length `n_causes`).
    pub n_cause_events: Vec<usize>,
    /// Local at-risk range of each row, in the order of the node's `rows`.
    local: Vec<LocalRow>,
}

impl NodeProfile {
    /// The criterion view of this node, given its distinct-unit count.
    pub fn view(&self, n_units: f64) -> Profile<'_> {
        Profile {
            at_risk: &self.at_risk,
            events: &self.events,
            cause_events: if self.n_causes == 1 {
                &self.events
            } else {
                &self.cause_events
            },
            n_causes: self.n_causes,
            times: &self.times,
            exposure: self.exposure,
            n_units,
        }
    }
}

/// Local `[la, lb)` at-risk range, cause code (0 = censored) and duration for each row of a node.
#[derive(Clone, Debug)]
struct LocalRow {
    la: u32,
    lb: u32,
    cause: u8,
    dur: f64,
}

/// Grid sizes up to this multiple of the node's row count use O(1) lookup
/// tables over the global grid; larger grids fall back to binary search.
const TABLE_FACTOR: usize = 4;

/// The node's event times (sorted global grid indices) and each row's local range.
fn node_event_idx_and_local(surv: &SurvData, rows: &[u32]) -> (Vec<u32>, Vec<LocalRow>) {
    let kg = surv.grid.len();
    if kg <= TABLE_FACTOR * rows.len().max(1) {
        // pos[g] = number of node event times below global index g.
        let mut pos = vec![0u32; kg + 1];
        for &r in rows {
            if surv.event[r as usize] != 0 {
                pos[surv.b[r as usize] as usize] = 1; // marks global index b - 1
            }
        }
        // Event index `e` is below global `g` iff its mark (at `e + 1`) is `<= g`.
        let mut event_idx = Vec::new();
        let mut run = 0u32;
        for (k, p) in pos.iter_mut().enumerate() {
            if *p == 1 {
                event_idx.push(k as u32 - 1);
                run += 1;
            }
            *p = run;
        }
        let local = rows
            .iter()
            .map(|&r| {
                let r = r as usize;
                LocalRow {
                    la: pos[surv.a[r] as usize],
                    lb: pos[surv.b[r] as usize],
                    cause: surv.event[r],
                    dur: surv.duration[r],
                }
            })
            .collect();
        return (event_idx, local);
    }
    let mut e: Vec<u32> = rows
        .iter()
        .filter(|&&r| surv.event[r as usize] != 0)
        .map(|&r| surv.b[r as usize] - 1)
        .collect();
    e.sort_unstable();
    e.dedup();
    let local = local_rows(surv, rows, &e);
    (e, local)
}

fn local_rows(surv: &SurvData, rows: &[u32], event_idx: &[u32]) -> Vec<LocalRow> {
    rows.iter()
        .map(|&r| {
            let r = r as usize;
            LocalRow {
                la: event_idx.partition_point(|&k| k < surv.a[r]) as u32,
                lb: event_idx.partition_point(|&k| k < surv.b[r]) as u32,
                cause: surv.event[r],
                dur: surv.duration[r],
            }
        })
        .collect()
}

/// `(at_risk, events, cause_events)` on `k` local event times; `cause_events`
/// is empty for `n_causes == 1`.
fn profile_from_local(
    local: &[LocalRow],
    k: usize,
    n_causes: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut diff = vec![0.0; k + 1];
    let mut events = vec![0.0; k];
    let mut cause_events = vec![0.0; if n_causes == 1 { 0 } else { k * n_causes }];
    for row in local {
        diff[row.la as usize] += 1.0;
        diff[row.lb as usize] -= 1.0;
        if row.cause != 0 {
            events[row.lb as usize - 1] += 1.0;
            if n_causes > 1 {
                cause_events[(row.cause as usize - 1) * k + row.lb as usize - 1] += 1.0;
            }
        }
    }
    let mut at_risk = vec![0.0; k];
    let mut run = 0.0;
    for j in 0..k {
        run += diff[j];
        at_risk[j] = run;
    }
    (at_risk, events, cause_events)
}

pub fn node_profile(surv: &SurvData, rows: &[u32]) -> NodeProfile {
    let (event_idx, local) = node_event_idx_and_local(surv, rows);
    let n_causes = surv.n_causes;
    let (at_risk, events, cause_events) = profile_from_local(&local, event_idx.len(), n_causes);
    let mut n_cause_events = vec![0usize; n_causes];
    for r in local.iter().filter(|r| r.cause != 0) {
        n_cause_events[r.cause as usize - 1] += 1;
    }
    let n_events = n_cause_events.iter().sum();
    let times = event_idx
        .iter()
        .map(|&k| surv.grid.times[k as usize])
        .collect();
    let exposure = local.iter().map(|r| r.dur).sum();
    NodeProfile {
        event_idx,
        at_risk,
        events,
        cause_events,
        n_causes,
        times,
        exposure,
        n_events,
        n_cause_events,
        local,
    }
}

#[derive(Clone, Debug)]
pub struct SplitParams {
    /// Minimum distinct units (ids, or bootstrap copies of ids) per child.
    /// An id whose rows fall on both sides counts in both children.
    pub min_leaf: usize,
    pub min_events_leaf: usize,
}

#[derive(Clone, Debug)]
pub struct SplitCandidate {
    pub feature: usize,
    /// Rows with `bin <= bin` go left.
    pub bin: u8,
    /// Raw-value threshold: `x <= threshold` goes left.
    pub threshold: f64,
    pub score: f64,
    /// Distinct units with at least one row in each child.
    pub ids_left: usize,
    pub ids_right: usize,
}

/// Number of distinct units, given that each unit's rows are contiguous.
pub fn count_units(units: &[u32]) -> usize {
    if units.is_empty() {
        return 0;
    }
    1 + units.windows(2).filter(|w| w[0] != w[1]).count()
}

/// Best admissible split over `features`. For each feature the left child
/// grows bin by bin; its at-risk profile is the prefix sum of one running
/// difference array over the node's event times.
///
/// `units[i]` is the resampling unit of `rows[i]`; each unit's rows must be
/// contiguous. Distinct-unit counts per child are exact: a unit is in the left
/// child iff its lowest bin is `<= b`, in the right iff its highest bin is `> b`.
///
/// Cost per feature: `O(n_node + bins_used * K_node)`; memory `O(n_node + K_node)`.
pub fn best_split(
    binned: &Binned,
    surv: &SurvData,
    rows: &[u32],
    units: &[u32],
    features: &[usize],
    params: &SplitParams,
    criterion: &dyn SplitCriterion,
) -> Option<SplitCandidate> {
    let profile = node_profile(surv, rows);
    best_split_in(binned, &profile, rows, units, features, params, criterion)
}

/// `best_split` for a node whose profile (of the same `rows`) is already known.
pub fn best_split_in(
    binned: &Binned,
    profile: &NodeProfile,
    rows: &[u32],
    units: &[u32],
    features: &[usize],
    params: &SplitParams,
    criterion: &dyn SplitCriterion,
) -> Option<SplitCandidate> {
    let k = profile.event_idx.len();
    if k == 0 {
        return None;
    }
    let local = &profile.local;
    debug_assert_eq!(rows.len(), local.len());
    debug_assert_eq!(rows.len(), units.len());
    let scorer = criterion.node_scorer(profile.view(count_units(units) as f64));
    let n_events = profile.n_events;
    let mut bins = vec![0u8; rows.len()];

    let mut best: Option<SplitCandidate> = None;
    let mut left_at = vec![0.0; k];
    let mut left_ev = vec![0.0; k];
    let nc = profile.n_causes;
    let mut left_cev = vec![0.0; if nc == 1 { 0 } else { k * nc }];
    let mut diff = vec![0.0; k + 1];
    let mut by_bin = vec![0u32; rows.len()];

    for &f in features {
        let col = binned.column(f);
        let mut counts = [0usize; 256];
        let mut ev_counts = [0usize; 256];
        // One gather of the node's bins; the passes below read them sequentially.
        for ((b, row), &r) in bins.iter_mut().zip(local).zip(rows) {
            *b = col[r as usize];
            counts[*b as usize] += 1;
            ev_counts[*b as usize] += (row.cause != 0) as usize;
        }
        let used: Vec<usize> = (0..256).filter(|&b| counts[b] > 0).collect();
        let nb = used.len();
        if nb < 2 {
            continue;
        }
        // Lowest and highest bin of each unit (runs of equal `units`).
        let (mut min_hist, mut max_hist) = ([0usize; 256], [0usize; 256]);
        let mut n_units = 0usize;
        let mut i = 0;
        while i < rows.len() {
            let unit = units[i];
            let (mut lo, mut hi) = (u8::MAX, 0u8);
            while i < rows.len() && units[i] == unit {
                let bin = bins[i];
                lo = lo.min(bin);
                hi = hi.max(bin);
                i += 1;
            }
            min_hist[lo as usize] += 1;
            max_hist[hi as usize] += 1;
            n_units += 1;
        }

        // Rows grouped by bin (counting sort), so the left child grows one bin
        // at a time with a single running difference array: O(K) memory.
        let mut offset = [0usize; 257];
        for b in 0..256 {
            offset[b + 1] = offset[b] + counts[b];
        }
        let mut fill = offset;
        for (i, &b) in bins.iter().enumerate() {
            by_bin[fill[b as usize]] = i as u32;
            fill[b as usize] += 1;
        }

        diff.iter_mut().for_each(|v| *v = 0.0);
        left_ev.iter_mut().for_each(|v| *v = 0.0);
        left_cev.iter_mut().for_each(|v| *v = 0.0);
        let (mut ids_left, mut max_le, mut e_left) = (0usize, 0usize, 0usize);
        let mut left_exposure = 0.0;
        for c in 0..nb - 1 {
            for &i in &by_bin[offset[used[c]]..offset[used[c] + 1]] {
                let row = &local[i as usize];
                left_exposure += row.dur;
                diff[row.la as usize] += 1.0;
                diff[row.lb as usize] -= 1.0;
                if row.cause != 0 {
                    left_ev[row.lb as usize - 1] += 1.0;
                    if nc > 1 {
                        left_cev[(row.cause as usize - 1) * k + row.lb as usize - 1] += 1.0;
                    }
                }
            }
            ids_left += min_hist[used[c]];
            max_le += max_hist[used[c]];
            e_left += ev_counts[used[c]];
            let (ids_right, e_right) = (n_units - max_le, n_events - e_left);
            if ids_left < params.min_leaf
                || ids_right < params.min_leaf
                || e_left < params.min_events_leaf
                || e_right < params.min_events_leaf
            {
                continue;
            }
            // Prefix sum only for admissible thresholds.
            let mut run = 0.0;
            for j in 0..k {
                run += diff[j];
                left_at[j] = run;
            }
            let left = Profile {
                at_risk: &left_at,
                events: &left_ev,
                cause_events: if nc == 1 { &left_ev } else { &left_cev },
                n_causes: nc,
                times: &profile.times,
                exposure: left_exposure,
                n_units: ids_left as f64,
            };
            let score = scorer.score(&left, ids_right as f64);
            if score > 0.0 && best.as_ref().is_none_or(|b| score > b.score) {
                let bin = used[c] as u8;
                best = Some(SplitCandidate {
                    feature: f,
                    bin,
                    threshold: binned.edges[f][bin as usize],
                    score,
                    ids_left,
                    ids_right,
                });
            }
        }
    }
    best
}

/// Person-time of `rows`.
pub fn exposure_of(surv: &SurvData, rows: &[u32]) -> f64 {
    rows.iter().map(|&r| surv.duration[r as usize]).sum()
}

/// Profile of `rows` evaluated on a given (e.g. parent's) event-time index set.
pub fn profile_on(surv: &SurvData, rows: &[u32], event_idx: &[u32]) -> (Vec<f64>, Vec<f64>) {
    let (at_risk, events, _) = cause_profile_on(surv, rows, event_idx);
    (at_risk, events)
}

/// `(at_risk, events, cause_events)` of `rows` on a given event-time index set;
/// `cause_events` is cause-major `[j * K + k]`, empty for `n_causes == 1`.
pub fn cause_profile_on(
    surv: &SurvData,
    rows: &[u32],
    event_idx: &[u32],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let local = local_rows(surv, rows, event_idx);
    profile_from_local(&local, event_idx.len(), surv.n_causes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::criterion::NodeScorer;
    use crate::grid::coarsen;
    use crate::rng::Rng;
    use std::sync::Mutex;

    /// What the splitter handed to a criterion for one candidate.
    #[derive(Debug)]
    struct Seen {
        at_risk: Vec<f64>,
        events: Vec<f64>,
        cause_events: Vec<f64>,
        n_causes: usize,
        times: Vec<f64>,
        exposure: f64,
        n_units: f64,
        n_units_right: f64,
    }

    /// Records the parent view and every scored left child, in order.
    #[derive(Default)]
    struct Spy {
        parent: Mutex<Option<Seen>>,
        children: Mutex<Vec<Seen>>,
    }

    fn seen(p: &Profile, n_units_right: f64) -> Seen {
        Seen {
            at_risk: p.at_risk.to_vec(),
            events: p.events.to_vec(),
            cause_events: p.cause_events.to_vec(),
            n_causes: p.n_causes,
            times: p.times.to_vec(),
            exposure: p.exposure,
            n_units: p.n_units,
            n_units_right,
        }
    }

    impl SplitCriterion for Spy {
        fn score(&self, _: &Profile, _: &Profile, _: f64) -> f64 {
            unreachable!("the splitter scores through node_scorer")
        }
        fn node_scorer<'a>(&'a self, parent: Profile<'a>) -> Box<dyn NodeScorer + 'a> {
            *self.parent.lock().unwrap() = Some(seen(&parent, f64::NAN));
            Box::new(SpyNode(self))
        }
    }

    struct SpyNode<'a>(&'a Spy);

    impl NodeScorer for SpyNode<'_> {
        fn score(&self, left: &Profile, n_units_right: f64) -> f64 {
            self.0
                .children
                .lock()
                .unwrap()
                .push(seen(left, n_units_right));
            0.0
        }
    }

    fn distinct(units: impl Iterator<Item = u32>) -> f64 {
        units.collect::<std::collections::HashSet<_>>().len() as f64
    }

    /// Every candidate's child summaries equal a brute-force recomputation:
    /// exposure (with delayed entry), at-risk/event counts (all causes and per
    /// cause), the parent's times, and distinct units on each side, including
    /// units whose rows straddle the threshold and children with no events.
    #[test]
    fn child_summaries_match_brute_force() {
        for n_causes in [1, 3] {
            child_summaries_match_brute_force_for(n_causes);
        }
    }

    fn child_summaries_match_brute_force_for(n_causes: usize) {
        let mut rng = Rng::new(23);
        let mut straddled = 0;
        let mut eventless = 0;
        for _ in 0..300 {
            let n = 2 + rng.below(50);
            let start: Vec<f64> = (0..n).map(|_| rng.below(6) as f64 * 0.5).collect();
            let stop: Vec<f64> = start
                .iter()
                .map(|s| s + 0.25 + rng.below(8) as f64 * 0.75)
                .collect();
            let codes: Vec<u8> = (0..n)
                .map(|_| {
                    if rng.below(3) == 0 {
                        1 + rng.below(n_causes) as u8
                    } else {
                        0
                    }
                })
                .collect();
            let x: Vec<f64> = (0..n).map(|_| rng.below(5) as f64).collect();
            // Sorted unit labels keep each unit's rows contiguous.
            let mut all_units: Vec<u32> = (0..n).map(|_| rng.below(n / 2 + 1) as u32).collect();
            all_units.sort_unstable();
            let surv = SurvData::with_causes(&start, &stop, &codes, n_causes);
            let binned = Binned::fit(&x, n, 1, 256);
            let rows: Vec<u32> = (0..n as u32).filter(|_| rng.below(4) > 0).collect();
            let units: Vec<u32> = rows.iter().map(|&r| all_units[r as usize]).collect();
            let parent = node_profile(&surv, &rows);
            let params = SplitParams {
                min_leaf: 0,
                min_events_leaf: 0,
            };
            let spy = Spy::default();
            best_split_in(&binned, &parent, &rows, &units, &[0], &params, &spy);
            if parent.event_idx.is_empty() {
                continue;
            }

            let p = spy.parent.lock().unwrap().take().unwrap();
            assert_eq!(p.exposure, exposure_of(&surv, &rows));
            assert_eq!(p.n_units, distinct(units.iter().copied()));
            let times: Vec<f64> = parent
                .event_idx
                .iter()
                .map(|&k| surv.grid.times[k as usize])
                .collect();
            assert_eq!(p.times, times);
            assert_eq!(p.n_causes, n_causes);
            let (_, p_ev, p_cev) = cause_profile_on(&surv, &rows, &parent.event_idx);
            assert_eq!(p.events, p_ev);
            assert_eq!(p.cause_events, if n_causes == 1 { p_ev } else { p_cev });
            for j in 0..n_causes {
                let want = rows
                    .iter()
                    .filter(|&&r| codes[r as usize] as usize == j + 1)
                    .count();
                assert_eq!(parent.n_cause_events[j], want);
            }

            let col = binned.column(0);
            let mut used: Vec<u8> = rows.iter().map(|&r| col[r as usize]).collect();
            used.sort_unstable();
            used.dedup();
            let children = spy.children.lock().unwrap();
            assert_eq!(children.len(), used.len().saturating_sub(1));
            for (c, &b) in children.iter().zip(&used) {
                let (left, right): (Vec<usize>, Vec<usize>) =
                    (0..rows.len()).partition(|&i| col[rows[i] as usize] <= b);
                let left_rows: Vec<u32> = left.iter().map(|&i| rows[i]).collect();
                let (at, ev, cev) = cause_profile_on(&surv, &left_rows, &parent.event_idx);
                assert_eq!(c.at_risk, at);
                assert_eq!(c.events, ev);
                // Brute force per cause, independent of `profile_from_local`.
                let want_cev: Vec<f64> = (0..parent.event_idx.len() * n_causes)
                    .map(|i| {
                        let (j, k) = (i / parent.event_idx.len(), i % parent.event_idx.len());
                        let t = parent.event_idx[k] as usize;
                        left_rows
                            .iter()
                            .filter(|&&r| {
                                surv.b[r as usize] as usize == t + 1
                                    && codes[r as usize] as usize == j + 1
                            })
                            .count() as f64
                    })
                    .collect();
                assert_eq!(c.cause_events, want_cev);
                if n_causes > 1 {
                    assert_eq!(cev, want_cev);
                }
                assert_eq!(c.times, times);
                let want = exposure_of(&surv, &left_rows);
                assert!((c.exposure - want).abs() <= 1e-12 * want.max(1.0));
                let (ul, ur) = (
                    distinct(left.iter().map(|&i| units[i])),
                    distinct(right.iter().map(|&i| units[i])),
                );
                assert_eq!((c.n_units, c.n_units_right), (ul, ur));
                straddled += (ul + ur > p.n_units) as usize;
                eventless += (ev.iter().sum::<f64>() == 0.0) as usize;
            }
        }
        assert!(straddled > 0 && eventless > 0, "{straddled} {eventless}");
    }

    /// In coarse mode the durations (and so exposure) are those of the
    /// snapped rows, not the original ones.
    #[test]
    fn coarse_exposure_uses_snapped_times() {
        let start = [0.0, 0.3, 1.2];
        let stop = [0.3, 1.7, 2.6];
        let event = [false, true, true];
        let c = coarsen(
            &start,
            &stop,
            &event,
            &[0, 1, 2],
            &[0, 3],
            &[0.0, 1.0, 2.0, 3.0],
        );
        // Rows snap to (0, 1], (1, 2], (2, 3]; the original first row (0, 0.3]
        // becomes (0, 1], so the coarse total differs from the original 2.6.
        assert_eq!(
            (c.start.clone(), c.stop.clone()),
            (vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0])
        );
        let surv = SurvData::new(&c.start, &c.stop, &c.event);
        assert_eq!(surv.duration, vec![1.0, 1.0, 1.0]);
        assert_eq!(node_profile(&surv, &[0, 1, 2]).exposure, 3.0);
    }

    /// The lookup-table path (small grids) equals sort + binary search on
    /// random delayed-entry data and random row subsets of any size.
    #[test]
    fn lookup_tables_match_binary_search() {
        let mut rng = Rng::new(11);
        for _ in 0..300 {
            let n = 1 + rng.below(60);
            let start: Vec<f64> = (0..n).map(|_| rng.below(6) as f64).collect();
            let stop: Vec<f64> = start
                .iter()
                .map(|s| s + 1.0 + rng.below(8) as f64)
                .collect();
            let event: Vec<bool> = (0..n).map(|_| rng.below(2) == 0).collect();
            let surv = SurvData::new(&start, &stop, &event);
            let rows: Vec<u32> = (0..n as u32).filter(|_| rng.below(3) > 0).collect();
            let (idx, local) = node_event_idx_and_local(&surv, &rows);
            let mut e: Vec<u32> = rows
                .iter()
                .filter(|&&r| surv.event[r as usize] != 0)
                .map(|&r| surv.b[r as usize] - 1)
                .collect();
            e.sort_unstable();
            e.dedup();
            assert_eq!(idx, e);
            for (a, b) in local.iter().zip(local_rows(&surv, &rows, &e)) {
                assert_eq!((a.la, a.lb, a.cause), (b.la, b.lb, b.cause));
            }
        }
    }
}
