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
    pub n_events: usize,
    /// Local at-risk range of each row, in the order of the node's `rows`.
    local: Vec<LocalRow>,
}

/// Local `[la, lb)` at-risk range and event flag for each row of a node.
#[derive(Clone, Debug)]
struct LocalRow {
    la: u32,
    lb: u32,
    event: bool,
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
            if surv.event[r as usize] {
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
                    event: surv.event[r],
                }
            })
            .collect();
        return (event_idx, local);
    }
    let mut e: Vec<u32> = rows
        .iter()
        .filter(|&&r| surv.event[r as usize])
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
                event: surv.event[r],
            }
        })
        .collect()
}

fn profile_from_local(local: &[LocalRow], k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut diff = vec![0.0; k + 1];
    let mut events = vec![0.0; k];
    for row in local {
        diff[row.la as usize] += 1.0;
        diff[row.lb as usize] -= 1.0;
        if row.event {
            events[row.lb as usize - 1] += 1.0;
        }
    }
    let mut at_risk = vec![0.0; k];
    let mut run = 0.0;
    for j in 0..k {
        run += diff[j];
        at_risk[j] = run;
    }
    (at_risk, events)
}

pub fn node_profile(surv: &SurvData, rows: &[u32]) -> NodeProfile {
    let (event_idx, local) = node_event_idx_and_local(surv, rows);
    let (at_risk, events) = profile_from_local(&local, event_idx.len());
    let n_events = local.iter().filter(|r| r.event).count();
    NodeProfile {
        event_idx,
        at_risk,
        events,
        n_events,
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
    let scorer = criterion.node_scorer(Profile {
        at_risk: &profile.at_risk,
        events: &profile.events,
    });
    debug_assert_eq!(rows.len(), units.len());
    let n_events = profile.n_events;
    let mut bins = vec![0u8; rows.len()];

    let mut best: Option<SplitCandidate> = None;
    let mut left_at = vec![0.0; k];
    let mut left_ev = vec![0.0; k];
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
            ev_counts[*b as usize] += row.event as usize;
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
        let (mut ids_left, mut max_le, mut e_left) = (0usize, 0usize, 0usize);
        for c in 0..nb - 1 {
            for &i in &by_bin[offset[used[c]]..offset[used[c] + 1]] {
                let row = &local[i as usize];
                diff[row.la as usize] += 1.0;
                diff[row.lb as usize] -= 1.0;
                if row.event {
                    left_ev[row.lb as usize - 1] += 1.0;
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
            };
            let score = scorer.score(&left);
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

/// Profile of `rows` evaluated on a given (e.g. parent's) event-time index set.
pub fn profile_on(surv: &SurvData, rows: &[u32], event_idx: &[u32]) -> (Vec<f64>, Vec<f64>) {
    let local = local_rows(surv, rows, event_idx);
    profile_from_local(&local, event_idx.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

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
                .filter(|&&r| surv.event[r as usize])
                .map(|&r| surv.b[r as usize] - 1)
                .collect();
            e.sort_unstable();
            e.dedup();
            assert_eq!(idx, e);
            for (a, b) in local.iter().zip(local_rows(&surv, &rows, &e)) {
                assert_eq!((a.la, a.lb, a.event), (b.la, b.lb, b.event));
            }
        }
    }
}
