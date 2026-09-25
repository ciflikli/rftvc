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
}

/// Local `[la, lb)` at-risk range and event flag for each row of a node.
struct LocalRow {
    la: u32,
    lb: u32,
    event: bool,
}

fn node_event_idx(surv: &SurvData, rows: &[u32]) -> Vec<u32> {
    let mut e: Vec<u32> = rows
        .iter()
        .filter(|&&r| surv.event[r as usize])
        .map(|&r| surv.b[r as usize] - 1)
        .collect();
    e.sort_unstable();
    e.dedup();
    e
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
    let event_idx = node_event_idx(surv, rows);
    let local = local_rows(surv, rows, &event_idx);
    let (at_risk, events) = profile_from_local(&local, event_idx.len());
    let n_events = local.iter().filter(|r| r.event).count();
    NodeProfile {
        event_idx,
        at_risk,
        events,
        n_events,
    }
}

#[derive(Clone, Debug)]
pub struct SplitParams {
    /// Minimum rows per child. With one row per id (S1) this equals ids;
    /// distinct-id counting for straddling ids arrives in S3.
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
    pub n_left: usize,
    pub n_right: usize,
}

/// Best admissible split over `features`, using per-feature histograms over
/// (bin x node event time) built with difference arrays.
///
/// Cost per feature: `O(n_node + bins_used * K_node)`.
pub fn best_split(
    binned: &Binned,
    surv: &SurvData,
    rows: &[u32],
    features: &[usize],
    params: &SplitParams,
    criterion: &dyn SplitCriterion,
) -> Option<SplitCandidate> {
    let event_idx = node_event_idx(surv, rows);
    let k = event_idx.len();
    if k == 0 {
        return None;
    }
    let local = local_rows(surv, rows, &event_idx);
    let (p_at, p_ev) = profile_from_local(&local, k);
    let parent = Profile {
        at_risk: &p_at,
        events: &p_ev,
    };
    let n = rows.len();
    let n_events = local.iter().filter(|r| r.event).count();

    let mut best: Option<SplitCandidate> = None;
    let mut left_at = vec![0.0; k];
    let mut left_ev = vec![0.0; k];
    let mut diff_run = vec![0.0; k + 1];

    for &f in features {
        let col = binned.column(f);
        let mut counts = [0usize; 256];
        let mut ev_counts = [0usize; 256];
        for (row, &r) in local.iter().zip(rows) {
            let bin = col[r as usize] as usize;
            counts[bin] += 1;
            ev_counts[bin] += row.event as usize;
        }
        let used: Vec<usize> = (0..256).filter(|&b| counts[b] > 0).collect();
        let nb = used.len();
        if nb < 2 {
            continue;
        }
        let mut compact = [0u16; 256];
        for (c, &b) in used.iter().enumerate() {
            compact[b] = c as u16;
        }

        let mut diff = vec![0.0; nb * (k + 1)];
        let mut ev = vec![0.0; nb * k];
        for (row, &r) in local.iter().zip(rows) {
            let c = compact[col[r as usize] as usize] as usize;
            diff[c * (k + 1) + row.la as usize] += 1.0;
            diff[c * (k + 1) + row.lb as usize] -= 1.0;
            if row.event {
                ev[c * k + row.lb as usize - 1] += 1.0;
            }
        }

        left_at.iter_mut().for_each(|v| *v = 0.0);
        left_ev.iter_mut().for_each(|v| *v = 0.0);
        let (mut n_left, mut e_left) = (0usize, 0usize);
        for c in 0..nb - 1 {
            diff_run.copy_from_slice(&diff[c * (k + 1)..(c + 1) * (k + 1)]);
            let mut run = 0.0;
            for j in 0..k {
                run += diff_run[j];
                left_at[j] += run;
                left_ev[j] += ev[c * k + j];
            }
            n_left += counts[used[c]];
            e_left += ev_counts[used[c]];
            let (n_right, e_right) = (n - n_left, n_events - e_left);
            if n_left < params.min_leaf
                || n_right < params.min_leaf
                || e_left < params.min_events_leaf
                || e_right < params.min_events_leaf
            {
                continue;
            }
            let left = Profile {
                at_risk: &left_at,
                events: &left_ev,
            };
            let score = criterion.score(&left, &parent);
            if score > 0.0 && best.as_ref().is_none_or(|b| score > b.score) {
                let bin = used[c] as u8;
                best = Some(SplitCandidate {
                    feature: f,
                    bin,
                    threshold: binned.edges[f][bin as usize],
                    score,
                    n_left,
                    n_right,
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
