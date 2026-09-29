"""
Pitcher Deception Project: called-strike outcome (validation table only, not part of Deception+)
=================================================================================================
A pitcher who gets more called strikes than their exact plate location alone would predict
may be stealing strikes through pitch shape, a different mechanism from whiff, chase, ground
ball, weak contact and the bat-tracking outcomes (which all need a swing or a ball in play).
It fails the 2025-to-2026 forecast and does not persist across seasons, so
reliability_and_ci.COMPOSITE_OUTCOMES leaves it out. It is still fit and scored.

Subset: pitches the batter DIDN'T swing at and that were a real ball/strike judgment call
(excludes hit_by_pitch and the pitch-clock-violation automatic_ball/automatic_strike calls,
which aren't about pitch shape at all). Target: description == "called_strike".

Fit at the full tier, reusing fit_full_model.py's `fit_full_outcome` (GroupKFold by pitcher,
batter+catcher tendency, count/park context), so it is the same methodology as the other outcomes.
"""

from build_pitch_table import PITCH_LEVEL_FILE, load_pitch_data, read_aligned_predictions, save_predictions
from fit_full_model import fit_full_outcome, prepare_context

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
    fit_full_outcome(df, "calledstrike", "is_called_strike", df["is_take"], "classify")
    save_predictions(existing, df, ["is_take", "is_called_strike", "calledstrike_expected_full"])
