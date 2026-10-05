"""
Pitcher Deception Project: called-strike outcome (validation table only, not part of Deception+)
=================================================================================================
A pitcher who gets more called strikes than their exact plate location alone would predict
may be stealing strikes through pitch shape, a different mechanism from whiff, chase, ground
ball, weak contact and the bat-tracking outcomes (which all need a swing or a ball in play).
It carries over weakly between seasons and adds little to the year-ahead forecast (it passes the p < 0.01
gate on three seasons with a gain far smaller than any member's), and adding it lowers the composite's
stability, so reliability_and_ci.COMPOSITE_OUTCOMES leaves it out. It is still fit and scored.

Subset: pitches the batter DIDN'T swing at and that were a real ball/strike judgment call
(excludes hit_by_pitch and the pitch-clock-violation automatic_ball/automatic_strike calls,
which aren't about pitch shape at all). Target: description == "called_strike".

Not month-recalibrated like the scored outcomes (fit_full_model.recalibrate_oof_by_group): its month gap
is small (SD 0.24 pp) and the recalibration cost log loss (0.1295 to 0.1300).

Fit at the full tier, reusing fit_full_model.py's `fit_full_outcome` (GroupKFold by pitcher,
batter+catcher tendency, count/park context), so it is the same methodology as the other outcomes.

Umpire identity was tested here as a candidate addition: MLB's public Stats API (unlike Statcast,
whose own `umpire` column ships empty) gives each game's home-plate umpire by game_pk, so an
umpire_tendency feature (the umpire's own called-strike rate on other pitchers' takes that season)
is buildable with the same fold-honest mechanism as batter and catcher tendency. A held-out test on
four-seamers found no real gain (mean AUC delta +0.0000, 95% interval -0.0000 to +0.0001, not every
fold positive): called strike's baseline AUC is already 0.988, almost entirely from plate location,
leaving little room for umpire identity to register. Not adopted; see README's Fixes section.
"""

from build_pitch_table import PITCH_LEVEL_FILE, load_pitch_data, read_aligned_predictions, save_predictions
from fit_full_model import FOLD_SEEDS, fit_full_outcome, prepare_context

TAKE_DESC = {"ball", "called_strike", "blocked_ball"}

if __name__ == "__main__":
    print("loading full pitch-level data...", flush=True)
    df = prepare_context(load_pitch_data(PITCH_LEVEL_FILE))
    df["is_take"] = df["description"].isin(TAKE_DESC)
    df["is_called_strike"] = (df["description"] == "called_strike").astype(int)
    print(f"takes: {df['is_take'].sum():,} / {len(df):,} pitches, "
          f"called-strike rate={df.loc[df['is_take'], 'is_called_strike'].mean():.3f}", flush=True)

    existing = read_aligned_predictions(df)

    print("\n=== CALLED STRIKE (full tier: stuff+location+batter+catcher+context) ===", flush=True)
    fit_full_outcome(df, "calledstrike", "is_called_strike", df["is_take"], "classify", fold_seeds=FOLD_SEEDS)
    save_predictions(existing, df, ["is_take", "is_called_strike", "calledstrike_expected_full"])
