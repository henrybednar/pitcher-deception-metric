"""
Pitcher Deception Project — split-half reliability + bootstrap CI + composite
=================================================================================
Reads per_pitch_predictions.csv (from build_pitch_table.py and the fit_* steps)
and, without refitting any model, computes for every outcome:

1. Full-season deception scores.
2. Split-half reliability (game_pk parity, Spearman-Brown corrected).
3. Bootstrap 95% CIs via a game-level cluster bootstrap.
4. A composite over the reliable components, correlation/PCA check, and a
   "qualified" tier.

Statistical fixes from the sabermetric audit, all applied here:

- A precision-weighted random-effects variance estimator (`estimate_true_var`)
  replaces the old `raw_var - mean(sampling_var)` naive subtraction, which
  implicitly treated every pitcher-season as equally precise (they range
  from n=30 to n=1,800+) and let a long tail of tiny-n, high-variance
  pitcher-seasons distort the population "true talent" variance estimate.
  It is the Paule-Mandel estimator with standardized residuals winsorized at
  WINSOR_Z, so a handful of extreme pitcher-seasons cannot inflate the
  between-pitcher variance that every other pitcher's shrinkage depends on.
  (DerSimonian-Laird, the earlier choice, let five saturated 30-pitch samples
  raise the estimate 16-fold in a stress test.)
- Binary outcomes use the pitch-level Bernoulli variance, sum of p(1-p) over the
  pitcher's own pitches, not the pooled p-bar(1-p-bar)/n. Predictions vary a lot
  within a pitcher-season, so the pooled formula overstated the variance by 1.3x
  (whiff), 1.5x (chase) and 4.8x (called strike).
- The design effect is the game-weighted ratio of the cluster-bootstrap variance
  to that analytic variance, with the (g-1)/g bootstrap bias undone. It used to be a median
  of per-pitcher ratios, which is biased low for short samples.
- Continuous outcomes (timing) now use a PITCH-TYPE-WEIGHTED sampling
  variance instead of one pooled constant across every pitch type — a
  four-seamer and a curveball don't have the same residual noise, so a
  pitcher's own pitch-type mix now determines their own sampling variance.
- "qualified" is computed HERE (not downstream) so the composite's scaling
  denominator (`pop_std`) can be computed over the QUALIFIED population
  instead of every `n_comp>=2` pitcher-season — a raw, unqualified outlier
  (see: the pre-fix "Thompson, Zach" case) was inflating the standard
  deviation that every OTHER pitcher's score gets normalized against.
- The composite is weighted by RELIABILITY x relative-n, not relative-n
  alone. Weighting purely by "how much of my own data do I have" let a
  pitcher's large sample in a noisy component (weak contact) count as much
  as an equally large sample in a reliable one (timing), and the composite's
  own reliability then fell below its best input component, which a
  well-built aggregate should not do.
- Binary sampling variance uses the predicted rate clipped to [0.01, 0.99] (BINARY_P_CLIP). About
  70 pitcher-seasons have a predicted rate of exactly 0 or 1, and the earlier floor of 1e-9 on the
  variance gave them a DerSimonian-Laird weight of 1e9. The estimated between-pitcher variance for
  whiff came out 24 times too large in one half of the data (understating split-half reliability,
  0.595 vs 0.664 once fixed), and chase's came out about twice too large.
- Shrinkage targets the precision-weighted grand mean (`center`), not zero, so a small calibration
  bias in the out-of-fold predictions moves the scale and not individual pitchers. Deception+ is
  centered on the qualified population, so 100 is the qualified average.
- Between-pitcher variance (tau2) is estimated from each pitcher collapsed to one point
  (`collapse_repeated_pitchers`), not from every pitcher-season row. About 650 pitchers have both a
  2025 and a 2026 row; leaving both in gives that pitcher's talent double the weight in tau2, since
  the two rows share a talent component instead of being independent draws of it. Measured on the
  current pull: tau2 comes out about 9% too high for both whiff and timing when left uncollapsed.
- Intervals are the wider of the game-cluster bootstrap and an analytic posterior interval
  (reliability x sampling variance). A saturated sample (30 whiffs out of 30) resamples to
  identical draws, which gave a zero-width bootstrap interval, and pitcher-seasons with fewer than
  3 games got no interval at all.
- Only pitches with a model expectation count toward n and the actual total. Pitch types outside
  the scored set are blanked upstream (pitch_hygiene.py), and rows without an expectation would
  otherwise add actual outcomes with no expected outcomes.
- `estimate_design_effect` reconciles a real inconsistency: bootstrap_ci
  treats games as clusters (pitches in a game aren't independent draws),
  but shrink_and_scale's sampling_var used to assume plain i.i.d. Bernoulli/
  per-pitch variance regardless — two different implicit data models in
  the same pipeline. Now a single empirically-estimated inflation factor
  (median ratio of bootstrap variance to the naive formula, per outcome)
  is applied consistently to both.

Also relevant, upstream: fit_full_model.py's batter/catcher tendency features
use leave-one-PITCHER-out mean encoding. A plain groupby().mean() includes each
pitch's own outcome in its own "opponent quality" feature, and leave-one-row-out
still lets a catcher's average re-encode the pitcher he usually catches.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2

from season_pairs import consecutive_pairs, correlation_by_pair, pooled_correlation

PITCH_LEVEL_FILE = "raw/statcast_pitch_level_2024_2026.csv"
FANGRAPHS_COLS = ["stuff_plus", "location_plus", "pitching_plus", "games", "games_started", "innings_pitched"]


def merge_fangraphs_columns(ps: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    """Left-merge whichever FANGRAPHS_COLS are present in covariates (pitcher x season); NaN for any
    that are absent, since merge_data.py's two FanGraphs merges (merge_fangraphs_stuff,
    merge_fangraphs_standard) each skip their columns entirely when that manual export hasn't been
    dropped in, rather than writing empty columns."""
    present = [c for c in FANGRAPHS_COLS if c in covariates.columns]
    ps = ps.merge(covariates[["pitcher", "season"] + present], on=["pitcher", "season"], how="left")
    for c in FANGRAPHS_COLS:
        if c not in ps.columns:
            ps[c] = np.nan
    return ps
MIN_N_FOR_SCORE = 30
N_BOOTSTRAP = 300
RNG = np.random.default_rng(42)  # bootstrap seed
# Floor for every population-std used as a scaling divisor (league_std,
# pop_std, std_h). A degenerate population (e.g. a filtering bug collapsing
# "qualified" to a handful of near-identical pitchers) would send std to 0
# and every score to +/-inf simultaneously — confirmed by direct injection.
# Doesn't change any real output (current populations have plenty of
# spread); it's a dormant-failure guard, not a live fix.
MIN_STD = 1e-6
# Each pitch's Bernoulli variance is p(1-p). A predicted rate of exactly 0 or 1 (GBM leaves can
# saturate) would give a variance of 0, and flooring that at 1e-9 once gave those rows a weight of
# 1e9 in the between-pitcher variance estimate. Clipping p keeps the variance realistic.
BINARY_P_CLIP = 0.01
# Between-pitcher variance is estimated on standardized residuals winsorized at this many
# standard deviations, and never goes below TAU2_FLOOR.
WINSOR_Z = 4.0
TAU2_FLOOR = 1e-9
# A component whose pitcher-seasons do not differ by more than sampling noise (Cochran's Q, winsorized,
# p above this) has nothing to rank, so every score is 100. Without the test, a component with no talent
# whose variance estimate escaped the floor still got an index spread of 10 (47% of 300 simulated
# no-talent populations).
HETEROGENEITY_ALPHA = 0.05
# A composite member whose league-wide split-half reliability is below this gets weight 0.
MIN_COMPONENT_RELIABILITY = 0.2

BINARY_OUTCOMES = {
    "whiff": dict(subset=lambda d: d["is_swing"], target="is_whiff"),
    "chase": dict(subset=lambda d: ~d["is_in_zone"], target="is_swing"),
    "gb": dict(subset=lambda d: d["is_bip"], target="is_gb"),
    "weak": dict(subset=lambda d: d["is_bip"], target="is_weak"),
    "calledstrike": dict(subset=lambda d: d["is_take"], target="is_called_strike"),
}
CONTINUOUS_OUTCOMES = {
    # Timing and alignment are scored on contact swings only. On a whiff the bat-ball intercept is a
    # closest-approach point, so whiffs carry large deviations by construction (14.0 in vs 7.3 in) and
    # would leak the whiff rate into the timing score. See fit_swing_alignment.py.
    "timing": dict(subset=lambda d: d["is_swing"] & (d["is_whiff"] == 0) & d["timing_dev_abs"].notna(), target="timing_dev_abs"),
    "align": dict(subset=lambda d: d["is_swing"] & (d["is_whiff"] == 0) & d["align_dev_abs"].notna(), target="align_dev_abs"),
    "whiffmiss": dict(subset=lambda d: (d["is_whiff"] == 1) & d["whiffmiss_log"].notna(), target="whiffmiss_log"),
}
# Every outcome here is scored, given intervals and tested. Only COMPOSITE_OUTCOMES feed
# Deception+, and each member is weighted by its league-wide split-half reliability.
# Membership is a decision made from the evidence, not an automatic rule. A candidate joins when it
# clears a split-half reliability of about 0.3 and the year-ahead forecast test (p below 0.01), and
# does not make the composite less stable. Weak contact is the exception on reliability (about 0.27):
# it is in on the forecast gate, and the composite is about as stable without it. The comparisons that
# back the decision (each member dropped, each candidate added, year-over-year over every pair of
# consecutive seasons with intervals) are produced by membership_check.py on every pipeline run and
# written up in the README and on the dashboard; on 2024-26 they say:
#   gb, calledstrike, align   each lowers the composite's year-over-year when added. Called strike and
#                             alignment now also pass the forecast gate, with gains a small fraction of a
#                             member's, so their exclusion rests on stability, not on the p-value.
#   whiff, whiffmiss          dropping either costs about 0.05
#   timing, weak, chase       dropping them costs about nothing or little, so they stay on their
#                             forecast gains and reliability
ALL_OUTCOMES = list(BINARY_OUTCOMES) + list(CONTINUOUS_OUTCOMES)
COMPOSITE_OUTCOMES = ["whiff", "chase", "weak", "timing", "whiffmiss"]
QUALIFY_MIN_N = {"whiff": 100, "chase": 100, "timing": 75, "gb": 40, "weak": 40, "calledstrike": 100,
                 "align": 75, "whiffmiss": 40}
QUALIFY_MIN = {f"{label}_n": QUALIFY_MIN_N[label] for label in COMPOSITE_OUTCOMES}


def estimate_true_var(diff: np.ndarray, sampling_var: np.ndarray, clip: float = WINSOR_Z) -> float:
    """Between-pitcher variance by the Paule-Mandel estimator: the tau2 at which the sum of squared
    standardized residuals equals k - 1, with weights 1 / (tau2 + sampling_var).

    Each standardized residual is winsorized at +/- clip before it is squared, so a few extreme
    pitcher-seasons (a saturated 30-pitch sample, or a real outlier talent) count as clip-sized
    misses and cannot dominate the estimate. If the residuals are already tighter than the sampling
    noise predicts, the answer is TAU2_FLOOR."""
    k = len(diff)

    def excess(tau2: float) -> float:
        w = 1.0 / (tau2 + sampling_var)
        mu = (w * diff).sum() / w.sum()
        z = np.clip((diff - mu) * np.sqrt(w), -clip, clip)
        return float((z ** 2).sum() - (k - 1))

    if k < 2 or excess(TAU2_FLOOR) <= 0:
        return TAU2_FLOOR
    hi = max(float(np.var(diff)), 10 * TAU2_FLOOR)
    while excess(hi) > 0 and hi < 1e6:
        hi *= 2
    return max(float(brentq(excess, TAU2_FLOOR, hi, xtol=1e-12, rtol=1e-10)), TAU2_FLOOR)


def has_between_pitcher_spread(diff: np.ndarray, sampling_var: np.ndarray, clip: float = WINSOR_Z,
                               alpha: float = HETEROGENEITY_ALPHA) -> bool:
    """Cochran's Q test, on winsorized standardized residuals, that pitcher-seasons differ by more than
    their sampling variances explain."""
    k = len(diff)
    if k < 2:
        return False
    w = 1.0 / sampling_var
    mu = (w * diff).sum() / w.sum()
    z = np.clip((diff - mu) * np.sqrt(w), -clip, clip)
    return bool(chi2.sf(float((z ** 2).sum()), k - 1) < alpha)


def collapse_repeated_pitchers(pitcher: np.ndarray, n: np.ndarray, diff: np.ndarray,
                               sampling_var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapses a pitcher's multiple pitcher-season rows to one n-weighted point, for estimating
    between-pitcher variance only. estimate_true_var and has_between_pitcher_spread both assume
    every row is an independent draw of "true talent"; two seasons from the SAME pitcher share that
    talent instead, so leaving both in gives that pitcher's talent twice the influence on tau2.
    Combining the two seasons' diffs by their own weights is exact variance algebra (the seasons'
    sampling noise IS independent), not an approximation."""
    frame = pd.DataFrame({"pitcher": pitcher, "n": n, "diff": diff, "sampling_var": sampling_var})

    def combine(g: pd.DataFrame) -> pd.Series:
        total_n = g["n"].sum()
        collapsed_diff = (g["diff"] * g["n"]).sum() / total_n
        collapsed_var = (g["sampling_var"] * (g["n"] / total_n) ** 2).sum()
        return pd.Series({"diff": collapsed_diff, "sampling_var": collapsed_var})

    collapsed = frame.groupby("pitcher").apply(combine, include_groups=False)
    return collapsed["diff"].to_numpy(), collapsed["sampling_var"].to_numpy()


def robust_std(x: np.ndarray, clip: float = WINSOR_Z) -> float:
    """Standard deviation after winsorizing at +/- clip robust SDs (1.4826 x the median absolute deviation)
    from the median. The index divisor uses this so a few extreme pitcher-seasons cannot stretch the
    scale for everyone: in a stress test five saturated samples tripled the plain SD."""
    med = float(np.median(x))
    spread = 1.4826 * float(np.median(np.abs(x - med)))
    if spread <= 0:
        return float(np.std(x))
    return float(np.std(np.clip(x, med - clip * spread, med + clip * spread)))


def binary_pitch_variance(expected: pd.Series) -> pd.Series:
    """Bernoulli variance p(1-p) of each pitch under its predicted probability, clipped so a saturated
    prediction does not give a zero-variance pitch."""
    p = expected.clip(BINARY_P_CLIP, 1 - BINARY_P_CLIP)
    return p * (1 - p)


def game_level_table(df: pd.DataFrame, label: str, target: str, baseline: str = "full",
                      var_col: str = None) -> pd.DataFrame:
    """pitcher x season x game_pk totals — the unit the bootstrap resamples.

    `baseline` picks which expectation column is the scoring baseline:
    "full" (stuff+location+opponent+catcher+context, the canonical
    Deception+ baseline) or "stuffloc" (stuff+location only — the
    timing-direction diagnostic, the one outcome with no "full" tier).

    `var_col`, when given (continuous outcomes only), is a per-pitch
    pitch-type-specific residual-variance estimate — summed here so the
    pitcher's own pitch-type mix determines their own sampling variance
    downstream, rather than one pooled constant across all pitch types.
    """
    agg_spec = dict(n=(target, "size"), actual_sum=(target, "sum"),
                     expected_sum=(f"{label}_expected_{baseline}", "sum"))
    if var_col is not None:
        agg_spec["var_sum"] = (var_col, "sum")
    return df.groupby(["pitcher", "season", "game_pk"]).agg(**agg_spec).reset_index()


def pitcher_season_point_estimate(game_tbl: pd.DataFrame) -> pd.DataFrame:
    """Season totals per pitcher. sampling_var is the summed pitch-level variance over n squared; it is
    NaN when game_tbl has no var_sum column (callers that only need the means)."""
    agg_spec = dict(n=("n", "sum"), actual_sum=("actual_sum", "sum"), expected_sum=("expected_sum", "sum"),
                     n_games=("game_pk", "nunique"))
    if "var_sum" in game_tbl.columns:
        agg_spec["var_sum"] = ("var_sum", "sum")
    agg = game_tbl.groupby(["pitcher", "season"]).agg(**agg_spec).reset_index()
    agg["actual_mean"] = agg["actual_sum"] / agg["n"]
    agg["expected_mean"] = agg["expected_sum"] / agg["n"]
    agg["diff"] = agg["actual_mean"] - agg["expected_mean"]
    if "var_sum" not in agg.columns:
        agg["sampling_var"] = np.nan
        return agg
    # A pitcher-season whose pitch-type residual variance is exactly zero would make sampling_var
    # literally 0 and blow up 1/sampling_var in the estimator. Floor it.
    agg["sampling_var"] = (agg["var_sum"] / (agg["n"] ** 2)).clip(lower=1e-9)
    return agg


def estimate_design_effect(game_tbl: pd.DataFrame, point_est: pd.DataFrame, n_boot: int = 200,
                            min_games: int = 5) -> float:
    """Ratio of empirical (game-cluster bootstrap) variance of `diff` to the
    analytic pitch-level `sampling_var`, pooled across pitcher-seasons with
    enough games: the sum over seasons of games x bootstrap variance divided by the
    sum of games x analytic variance. A bootstrap over g clusters understates the
    variance by (g-1)/g, so each bootstrap variance is scaled by g/(g-1) first.

    bootstrap_ci already treats games as clusters (pitches within a game
    aren't independent draws) — but shrink_and_scale's sampling_var used
    the naive analytic formula unconditionally, an inconsistency between
    two parts of the same pipeline that implicitly assumed two different
    data-generating processes. >1 means pitches are positively correlated
    within a game (the expected direction — hot/cold outings, sequencing,
    matchup effects), so the naive formula understated true noise and
    shrinkage was too weak, not too strong. Floored at 1.0: a noisy
    estimate should never be allowed to REDUCE variance below the already-
    conservative analytic baseline.
    """
    pv = point_est.set_index(["pitcher", "season"])["sampling_var"]
    weighted_empirical = weighted_analytic = 0.0
    for (pitcher, season), g in game_tbl.groupby(["pitcher", "season"]):
        if len(g) < min_games:
            continue
        key = (pitcher, season)
        if key not in pv.index:
            continue
        analytic_var = pv.loc[key]
        if not np.isfinite(analytic_var) or analytic_var <= 0:
            continue
        n_arr, a_arr, e_arr = g["n"].values, g["actual_sum"].values, g["expected_sum"].values
        idx = RNG.integers(0, len(g), size=(n_boot, len(g)))
        boot_n = n_arr[idx].sum(axis=1)
        boot_diff = (a_arr[idx].sum(axis=1) - e_arr[idx].sum(axis=1)) / boot_n
        empirical_var = boot_diff.var() * len(g) / (len(g) - 1)
        if np.isfinite(empirical_var):
            weighted_empirical += len(g) * empirical_var
            weighted_analytic += len(g) * analytic_var
    return max(weighted_empirical / weighted_analytic, 1.0) if weighted_analytic > 0 else 1.0


def shrink_and_scale(agg: pd.DataFrame, design_effect: float = 1.0):
    """Empirical-Bayes shrinkage and rescale to 100/10.
    Returns (shrunk_diff, index, league_std, true_var, center).

    Shrinkage pulls each pitcher-season toward the precision-weighted grand mean `center`, not toward
    zero, so a small calibration bias in the out-of-fold predictions (whiff runs +0.0019) moves the
    whole scale and not every pitcher. `shrunk_diff` is the deviation from that center, so index 100
    is the average scored pitcher-season.

    `design_effect` (see estimate_design_effect) inflates the naive per-observation sampling_var to
    account for within-game correlation. Pitches in the same outing aren't independent draws, the
    same reason the CI bootstrap resamples whole games rather than individual pitches.
    """
    scored = agg[agg["n"] >= MIN_N_FOR_SCORE].copy()
    sampling_var = scored["sampling_var"].values * design_effect
    diff_c, sampling_var_c = collapse_repeated_pitchers(
        scored["pitcher"].values, scored["n"].values, scored["diff"].values, sampling_var)
    if has_between_pitcher_spread(diff_c, sampling_var_c):
        true_var = estimate_true_var(diff_c, sampling_var_c)
    else:
        true_var = TAU2_FLOOR
    weights = 1.0 / (true_var + sampling_var)
    center = float(np.sum(weights * scored["diff"].values) / np.sum(weights))
    reliability = true_var / (true_var + sampling_var)
    shrunk = (scored["diff"].values - center) * reliability
    if true_var <= TAU2_FLOOR:
        shrunk = np.zeros_like(shrunk)
        league_std = MIN_STD
    else:
        league_std = max(robust_std(shrunk), MIN_STD)

    out_shrunk = pd.Series(np.nan, index=agg.index)
    out_shrunk.loc[scored.index] = shrunk
    out_index = 100 + 10 * out_shrunk / league_std
    return out_shrunk, out_index, league_std, true_var, center


def posterior_sd_z(sampling_var, design_effect: float, true_var: float, league_std: float) -> np.ndarray:
    """Posterior standard deviation of a component's estimate on the z scale (index 10 points = 1 z).
    The normal-normal posterior variance is true_var x s2 / (true_var + s2), reliability x sampling variance."""
    s2 = np.asarray(sampling_var, dtype=float) * design_effect
    return np.sqrt(true_var * s2 / (true_var + s2)) / league_std


def analytic_interval(agg: pd.DataFrame, index: pd.Series, league_std: float, true_var: float,
                      design_effect: float) -> pd.DataFrame:
    """95% posterior interval for the empirical-Bayes estimate, on the index scale.

    The posterior variance of a normal-normal estimate is reliability x sampling variance, so the
    interval is never zero-width, even for a saturated sample (30 whiffs out of 30) where every
    bootstrap resample is identical. It also exists for pitcher-seasons with fewer than 3 games,
    which the bootstrap skips."""
    half = 1.96 * 10 * posterior_sd_z(agg["sampling_var"], design_effect, true_var, league_std)
    return pd.DataFrame({"pitcher": agg["pitcher"], "season": agg["season"],
                         "an_lo": index - half, "an_hi": index + half})


def bootstrap_ci(game_tbl: pd.DataFrame, league_std: float, true_var: float,
                  design_effect: float = 1.0, min_n: int = MIN_N_FOR_SCORE, center: float = 0.0):
    """Game-level cluster bootstrap CI, vectorized per pitcher-season. Uses
    the SAME sampling-variance formula as the point estimate (summed pitch-level
    variance, scaled by the same design_effect used in shrink_and_scale),
    recomputed per bootstrap draw since resampled games change both n and the
    pitch mix."""
    results = []
    for (pitcher, season), g in game_tbl.groupby(["pitcher", "season"]):
        n_arr = g["n"].values
        a_arr = g["actual_sum"].values
        e_arr = g["expected_sum"].values
        v_arr = g["var_sum"].values
        n_games = len(g)
        if n_games < 3:
            continue
        idx = RNG.integers(0, n_games, size=(N_BOOTSTRAP, n_games))
        boot_n = n_arr[idx].sum(axis=1)
        boot_a = a_arr[idx].sum(axis=1)
        boot_e = e_arr[idx].sum(axis=1)
        valid = boot_n >= min_n
        if valid.sum() < N_BOOTSTRAP * 0.5:
            continue
        boot_diff = (boot_a[valid] - boot_e[valid]) / boot_n[valid]

        boot_v = v_arr[idx].sum(axis=1)[valid]
        samp_var = boot_v / (boot_n[valid] ** 2) * design_effect
        reliability = true_var / (true_var + samp_var)
        boot_shrunk = (boot_diff - center) * reliability
        boot_index = 100 + 10 * boot_shrunk / league_std

        lo, hi = np.percentile(boot_index, [2.5, 97.5])
        results.append({"pitcher": pitcher, "season": season, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(results)


def composite_weights(frame: pd.DataFrame, index_cols: list, n_cols: list, reliabilities: list) -> np.ndarray:
    """Composite weights, rows x components, zero where a component is absent or too unreliable.
    Each is the component's league-wide reliability times the pitcher's n over that component's median n."""
    w = np.zeros((len(frame), len(index_cols)))
    for i, (idx_col, n_col, r) in enumerate(zip(index_cols, n_cols, reliabilities)):
        median_n = frame.loc[frame[idx_col].notna(), n_col].median()
        rel_weight = r if pd.notna(r) and r >= MIN_COMPONENT_RELIABILITY else 0.0
        w[:, i] = np.where(frame[idx_col].notna(), rel_weight * frame[n_col] / median_n, 0.0)
    return w


def composite_posterior_sd(weights: np.ndarray, component_sd_z: np.ndarray) -> np.ndarray:
    """Posterior sd of the weighted-average composite z, treating the components' errors as independent."""
    weighted = np.nan_to_num(weights * component_sd_z)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt((weighted ** 2).sum(axis=1)) / weights.sum(axis=1)


def composite_interval(deception_plus: np.ndarray, composite_sd_z: np.ndarray, pop_std: float):
    """95% interval on the Deception+ scale. A composite z of 1 is `pop_std` raw units, scaled to 10 points."""
    half = 1.96 * 10 * composite_sd_z / pop_std
    return deception_plus - half, deception_plus + half


def compute_composite(frame: pd.DataFrame, index_cols: list, n_cols: list, reliabilities: list):
    """Composite z weighted by BOTH (a) each component's own population-level reliability, so a
    fundamentally noisy component (weak contact) can't count as much as a reliable one (timing)
    regardless of sample size, with components under MIN_COMPONENT_RELIABILITY getting weight 0, and
    (b) each pitcher's own n relative to what's typical for that specific component, so whiff/chase
    (many more opportunities per season) don't swamp ground ball/weak contact just because they have
    bigger raw counts. Returns (raw composite z, components present per row, eligible rows)."""
    z = (frame[index_cols].values - 100) / 10
    w = composite_weights(frame, index_cols, n_cols, reliabilities)
    n_comp = (~np.isnan(z)).sum(axis=1)
    z_filled = np.nan_to_num(z, nan=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_z = (z_filled * w).sum(axis=1) / w.sum(axis=1)
    eligible = n_comp >= 2
    raw_z = np.where(eligible, raw_z, np.nan)
    return raw_z, n_comp, eligible


def process_outcome(df: pd.DataFrame, label: str, spec: dict, is_binary: bool, baseline: str = "full"):
    # Only pitches the model actually predicted. Rows with no expectation (blanked pitch types) would
    # otherwise count toward n and the actual total but add nothing to the expected total.
    mask = spec["subset"](df) & df[f"{label}_expected_{baseline}"].notna()
    target = spec["target"]
    sub = df[mask].copy()

    # Per-pitch sampling variance, summed into each pitcher-season. Binary outcomes: the Bernoulli
    # variance of the pitch's own prediction. Continuous outcomes: the per-pitch-type residual variance,
    # since a four-seamer and a curveball don't have the same unexplained noise.
    if is_binary:
        sub["_var_contrib"] = binary_pitch_variance(sub[f"{label}_expected_{baseline}"])
    else:
        resid = sub[target] - sub[f"{label}_expected_{baseline}"]
        sub["_var_contrib"] = (resid ** 2).groupby(sub["pitch_type"]).transform("mean")
    var_col = "_var_contrib"

    # --- full season ---
    full_games = game_level_table(sub, label, target, baseline, var_col)
    full_agg = pitcher_season_point_estimate(full_games)
    # Calibrate ONCE from the full season (a stable estimate) and reuse the
    # same scalar for both halves and the bootstrap — consistent with how
    # the composite's reliability weights are also fixed from the full
    # season rather than re-derived on smaller, noisier half-samples.
    design_effect = estimate_design_effect(full_games, full_agg)
    full_shrunk, full_index, league_std, true_var, center = shrink_and_scale(full_agg, design_effect)
    full_agg[f"{label}_diff_adj"] = full_agg["diff"]
    full_agg[f"{label}_diff_adj_shrunk"] = full_shrunk
    full_agg[f"{label}_index"] = full_index
    full_agg = full_agg.rename(columns={"n": f"{label}_n"})
    full_agg[f"{label}_post_sd_z"] = posterior_sd_z(full_agg["sampling_var"], design_effect, true_var, league_std)

    # --- split half ---
    half_scores = {}
    for h in (0, 1):
        h_games = game_level_table(sub[sub["half"] == h], label, target, baseline, var_col)
        h_agg = pitcher_season_point_estimate(h_games)
        h_shrunk, h_index, *_ = shrink_and_scale(h_agg, design_effect)
        h_agg[f"{label}_index_h{h}"] = h_index
        half_scores[h] = h_agg[["pitcher", "season", f"{label}_index_h{h}", "n"]].rename(
            columns={"n": f"{label}_n_h{h}"})

    half_merged = half_scores[0].merge(half_scores[1], on=["pitcher", "season"])
    reliable = half_merged.dropna(subset=[f"{label}_index_h0", f"{label}_index_h1"])
    r_half = reliable[f"{label}_index_h0"].corr(reliable[f"{label}_index_h1"])
    r_full_sb = 2 * r_half / (1 + r_half) if pd.notna(r_half) and r_half > -1 else np.nan

    # --- bootstrap CI ---
    # The interval is the wider of the game-cluster bootstrap and the analytic posterior interval. The
    # bootstrap captures within-game clustering, the analytic one never collapses to zero width and
    # covers pitcher-seasons with fewer than 3 games.
    ci = bootstrap_ci(full_games, league_std, true_var, design_effect, center=center)
    an = analytic_interval(full_agg, full_agg[f"{label}_index"], league_std, true_var, design_effect)
    full_agg = full_agg.merge(ci, on=["pitcher", "season"], how="left").merge(an, on=["pitcher", "season"], how="left")
    full_agg["ci_lo"] = np.fmin(full_agg["ci_lo"], full_agg["an_lo"])
    full_agg["ci_hi"] = np.fmax(full_agg["ci_hi"], full_agg["an_hi"])
    full_agg = full_agg.rename(columns={"ci_lo": f"{label}_ci_lo", "ci_hi": f"{label}_ci_hi"})

    print(f"{label}: n_scored={len(full_agg.dropna(subset=[f'{label}_index']))}, "
          f"n_half_reliable={len(reliable)}, split-half r={r_half:.3f}, "
          f"Spearman-Brown corrected full-season r={r_full_sb:.3f}, design_effect={design_effect:.2f}", flush=True)

    return full_agg[["pitcher", "season", f"{label}_n", f"{label}_diff_adj", f"{label}_diff_adj_shrunk",
                      f"{label}_index", f"{label}_ci_lo", f"{label}_ci_hi", f"{label}_post_sd_z"]], {
        "label": label, "r_half": r_half, "r_full_spearman_brown": r_full_sb, "n_half_reliable": len(reliable),
        "design_effect": design_effect,
    }, half_merged


if __name__ == "__main__":
    print("loading per-pitch predictions...", flush=True)
    df = pd.read_csv("output/per_pitch_predictions.csv")
    print(f"rows: {len(df):,}", flush=True)

    ps = None
    half_tables = []
    reliability_report = []
    component_reliability = {}
    for label, spec in BINARY_OUTCOMES.items():
        result, rel, half_tbl = process_outcome(df, label, spec, is_binary=True)
        reliability_report.append(rel)
        half_tables.append(half_tbl)
        component_reliability[label] = rel["r_full_spearman_brown"]
        ps = result if ps is None else ps.merge(result, on=["pitcher", "season"], how="outer")
    for label, spec in CONTINUOUS_OUTCOMES.items():
        result, rel, half_tbl = process_outcome(df, label, spec, is_binary=False)
        reliability_report.append(rel)
        half_tables.append(half_tbl)
        component_reliability[label] = rel["r_full_spearman_brown"]
        ps = ps.merge(result, on=["pitcher", "season"], how="outer")

    half_all = half_tables[0]
    for t in half_tables[1:]:
        half_all = half_all.merge(t, on=["pitcher", "season"], how="outer")

    # --- qualified tier (computed here so the composite's own scaling can
    # use it — see module docstring) ---
    qualified = pd.Series(True, index=ps.index)
    for col, min_n in QUALIFY_MIN.items():
        qualified &= ps[col].fillna(0) >= min_n
    ps["qualified"] = qualified

    # --- correlation / PCA check ---
    z_cols = []
    for label in ALL_OUTCOMES:
        z_col = f"{label}_z"
        ps[z_col] = (ps[f"{label}_index"] - 100) / 10
        z_cols.append(z_col)

    corr = ps[z_cols].corr(min_periods=200)
    print(f"\ncorrelation matrix (all {len(ALL_OUTCOMES)} scored outcomes):\n", corr.round(3), flush=True)
    composite_z_cols = [f"{label}_z" for label in COMPOSITE_OUTCOMES]
    valid = ps.dropna(subset=composite_z_cols)
    eigvals, eigvecs = np.linalg.eigh(valid[composite_z_cols].corr().values)
    order = np.argsort(eigvals)[::-1]
    print(f"\nn with all {len(COMPOSITE_OUTCOMES)} composite members:{len(valid)}, eigenvalues: {eigvals[order].round(3)}, "
          f"PC1 variance explained: {(eigvals[order][0]/len(COMPOSITE_OUTCOMES)*100):.1f}%", flush=True)
    print("PC1 loadings:", dict(zip(COMPOSITE_OUTCOMES, eigvecs[:, order[0]].round(3))), flush=True)

    # --- composite, reliability x relative-n weighted ---

    composite_reliabilities = [component_reliability[l] for l in COMPOSITE_OUTCOMES]
    print(f"\ncomposite component reliabilities (used as fixed weights): "
          f"{dict(zip(COMPOSITE_OUTCOMES, [round(r, 3) for r in composite_reliabilities]))}", flush=True)

    raw_composite_z, n_components, eligible = compute_composite(
        ps, [f"{l}_index" for l in COMPOSITE_OUTCOMES], [f"{l}_n" for l in COMPOSITE_OUTCOMES],
        composite_reliabilities,
    )
    ps["deception_n_components"] = n_components
    # Scale against the QUALIFIED population, not every n_comp>=2 row — an
    # unqualified small-sample outlier shouldn't be able to inflate the SD
    # that every pitcher's score (qualified or not) gets normalized against.
    scaling_pop = eligible & ps["qualified"].values
    pop = pd.Series(raw_composite_z).loc[scaling_pop]
    pop_mean, pop_std = float(pop.mean()), max(float(pop.std()), MIN_STD)
    ps["deception_plus"] = np.where(eligible, 100 + 10 * (raw_composite_z - pop_mean) / pop_std, np.nan)

    # Interval: the components' posterior variances combined with the composite weights, assuming the
    # components' errors are independent. Weights are fixed at their point values.
    member_idx, member_n = [f"{l}_index" for l in COMPOSITE_OUTCOMES], [f"{l}_n" for l in COMPOSITE_OUTCOMES]
    weights = composite_weights(ps, member_idx, member_n, composite_reliabilities)
    sd_z = composite_posterior_sd(weights, ps[[f"{l}_post_sd_z" for l in COMPOSITE_OUTCOMES]].values)
    lo, hi = composite_interval(ps["deception_plus"].values, np.where(eligible, sd_z, np.nan), pop_std)
    ps["deception_plus_ci_lo"], ps["deception_plus_ci_hi"] = lo, hi

    # --- composite split-half reliability, same weights (fixed from the
    # full-season estimate, not re-derived per half) applied to each half ---
    for h in (0, 1):
        z_h, _, elig_h = compute_composite(
            half_all, [f"{l}_index_h{h}" for l in COMPOSITE_OUTCOMES], [f"{l}_n_h{h}" for l in COMPOSITE_OUTCOMES],
            composite_reliabilities,
        )
        qualified_lookup = ps.set_index(["pitcher", "season"])["qualified"]
        half_qualified = half_all.set_index(["pitcher", "season"]).index.map(qualified_lookup).fillna(False).values
        std_h = max(pd.Series(z_h).loc[elig_h & half_qualified].std(), MIN_STD)
        half_all[f"deception_plus_h{h}"] = np.where(elig_h, 100 + 10 * z_h / std_h, np.nan)

    comp_half_reliable = half_all.dropna(subset=["deception_plus_h0", "deception_plus_h1"])
    r_comp_half = comp_half_reliable["deception_plus_h0"].corr(comp_half_reliable["deception_plus_h1"])
    r_comp_full_sb = 2 * r_comp_half / (1 + r_comp_half) if pd.notna(r_comp_half) and r_comp_half > -1 else np.nan
    print(f"\nDeception+ split-half: n={len(comp_half_reliable)}, r={r_comp_half:.3f}, "
          f"Spearman-Brown corrected full-season r={r_comp_full_sb:.3f}", flush=True)
    reliability_report.append({"label": "deception_plus", "r_half": r_comp_half,
                                "r_full_spearman_brown": r_comp_full_sb, "n_half_reliable": len(comp_half_reliable)})

    comp_scored = ps.dropna(subset=["deception_plus"])
    print(f"\nDeception+ ({len(COMPOSITE_OUTCOMES)}-component) computed for {len(comp_scored):,} pitcher-seasons.", flush=True)
    print(f"component coverage: {comp_scored['deception_n_components'].value_counts().sort_index().to_dict()}", flush=True)
    print(f"qualified: {int(ps['qualified'].sum()):,}", flush=True)

    # Columns that do not depend on the model: player names, and the FanGraphs figures merged by
    # merge_data.py. FanGraphs is never used in scoring, only for the dashboard's Stuff+ axis and the
    # leaderboard column.
    names = pd.read_csv(PITCH_LEVEL_FILE, usecols=["pitcher", "player_name"]).drop_duplicates("pitcher")
    covariates = pd.read_csv("output/pitcher_season_covariates.csv")
    ps = ps.merge(names, on="pitcher", how="left")
    ps = merge_fangraphs_columns(ps, covariates)
    pairs = consecutive_pairs(comp_scored, ["deception_plus"])
    print(f"Deception+ year-over-year reliability: n={len(pairs)} pairs, r={pooled_correlation(pairs, 'deception_plus'):.3f} "
          f"({', '.join(f'{k} {v:.3f}' for k, v in correlation_by_pair(pairs, 'deception_plus').items())})", flush=True)

    top = ps.dropna(subset=["deception_plus"]).sort_values("deception_plus", ascending=False)
    print("\nTop 10 by Deception+ (with 95% CI on whiff component):")
    print(top.head(10)[["player_name", "season", "deception_plus", "qualified", "whiff_index", "whiff_ci_lo", "whiff_ci_hi",
                          "timing_index", "timing_ci_lo", "timing_ci_hi"]].to_string(index=False))

    ps.to_csv("output/pitcher_season.csv", index=False)
    pd.DataFrame(reliability_report).to_csv("output/reliability_report.csv", index=False)
    half_all.to_csv("output/half_scores.csv", index=False)      # every outcome's odd-game and even-game index, read by membership_check.py
    print("\nSaved pitcher_season.csv, reliability_report.csv and half_scores.csv. Done.", flush=True)
