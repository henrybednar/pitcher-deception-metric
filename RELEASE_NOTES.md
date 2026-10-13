# Deception+ v1.0.0

First public release of Deception+, a pitcher metric built from Statcast pitch-level data (2024 to 2026, through September 27, 2026). It scores how far a pitcher's results land above or below what a model predicts from each pitch's measured traits, the batter, the catcher, the count and the park. 100 is the average qualified pitcher and 10 points is one standard deviation.

Live: [dashboard](https://henrybednar.github.io/pitcher-deception-metric/) · [leaderboard](https://henrybednar.github.io/pitcher-deception-metric/deception_leaderboard.html) · [methodology](https://henrybednar.github.io/pitcher-deception-metric/deception_methodology.html)

## What is in it

- A reproducible pipeline (`python run_pipeline.py`): out-of-fold gradient-boosted pitch models, empirical-Bayes shrinkage, bootstrap and analytic intervals, and a check that every README headline number matches the outputs.
- A dashboard that walks one pitcher-season through the method, a stats key, and the validation (reliability, forecasts, on-field results).
- A sortable leaderboard of every scored pitcher-season, with pitch-type scores and 95% intervals.
- A methodology page with the model validation, the known limits and an audit log of the leaks and bugs found along the way.
- 340 tests, run on every push.

## Results at release

- Reliability r=0.741 within a season and r=0.611 from one season to the next.
- Pitch models rank whiffs and chases well (AUC 0.76 to 0.86 and 0.82 to 0.89) and are calibrated.
- It correlates r=0.386 with Stuff+, so it is not independent of pitch quality.
- Next season, it adds clearly to Stuff+ and Location+ for strikeout rate and xwOBA allowed.

## Known limits

- It is whatever the model cannot explain, not proof of deception.
- Three seasons of data: no aging curves, and two season-to-season pairs.
- The typical 95% interval is about 13 points, so only 22% of qualified scores are clearly different from 100.

## Data

Pitch data is from Baseball Savant (Statcast). Stuff+ is from FanGraphs, entered by hand, and is never used by the model. This project is independent of MLB, Baseball Savant and FanGraphs.
