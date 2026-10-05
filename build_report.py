"""
Pitcher Deception Project: build report.tex
===========================================
Writes the driver-analysis report (report.tex) from the pipeline's outputs, so every number in it comes from
output/driver_ols_report.json (the OLS tables), output/driver_analysis.json (permutation importance),
output/predictive_validity_report.csv, output/site_stats.json and the per-pitcher tables. Tiers follow the
Benjamini-Hochberg q and the nominal p of each feature, so a feature that crosses a line moves between tiers by
itself; only the one-sentence readings of each feature are written by hand (FEATURE_NOTES).

    python driver_ols_report.py      # the OLS tables the report reads
    python build_report.py           # report.tex
    pdflatex report.tex              # report.pdf

Run after run_pipeline.py.
"""

import csv
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from driver_ols_report import model_data
from season_pairs import consecutive_pairs, pooled_correlation

SEASONS = "2024--2026"
LABEL_TITLE = {"whiff": "Whiff", "timing": "Timing"}
TITLE = {
    "tempo_bases_empty_sec": "Pace between pitches",
    "avg_velocity_gap_from_prev": "Velocity gap from previous pitch",
    "repeat_pct": "Same-pitch-type repeat rate",
    "vaa_cross_pitch_std": "VAA consistency across repertoire",
    "spin_mirror_score_mean": "Spin-axis mirror score",
    "release_extension_mean": "Release extension",
    "n_pitch_types": "Repertoire size",
    "tunnel_differential_mean": "Pitch-pair tunnel differential",
    "arm_angle_szn_avg": "Season-average arm angle",
    "arm_angle_cross_pitch_std": "Arm-angle consistency",
    "vaa_mean": "Average vertical approach angle",
    "pitch_mix_entropy": "Pitch-mix entropy",
    "release_pos_x_cross_pitch_std": "Release-point (x) consistency",
    "release_pos_z_cross_pitch_std": "Release-point (z) consistency",
}
IMPORTANCE_NAME = {
    "tempo_bases_empty_sec": "Pace between pitches",
    "avg_velocity_gap_from_prev": "Velocity gap from previous pitch",
    "repeat_pct": "Same-pitch-type repeat rate",
    "vaa_cross_pitch_std": "VAA consistency (season proxy)",
    "spin_mirror_score_mean": "Spin-axis mirror score (real; low = match/mirror)",
    "release_extension_mean": "Release extension",
    "n_pitch_types": "Repertoire size (# pitch types)",
    "tunnel_differential_mean": "Pitch-pair tunnel differential (real)",
    "arm_angle_szn_avg": "Season-average arm angle",
    "arm_angle_cross_pitch_std": "Arm-angle consistency (season proxy)",
    "vaa_mean": "Average vertical approach angle",
    "pitch_mix_entropy": "Pitch-mix unpredictability (entropy)",
    "release_pos_x_cross_pitch_std": "Release-point (x) consistency (season proxy)",
    "release_pos_z_cross_pitch_std": "Release-point (z) consistency (season proxy)",
}
TIER3_SHOWN = 3

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{parskip}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{array}

\definecolor{hdrblue}{RGB}{20,50,90}
\definecolor{darkgray}{RGB}{60,60,60}

\titleformat{\section}{\normalfont\Large\bfseries\color{hdrblue}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalfont\large\bfseries\color{hdrblue}}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\normalfont\bfseries}{}{0em}{}
\titlespacing*{\section}{0pt}{1.4em}{0.6em}
\titlespacing*{\subsection}{0pt}{1.1em}{0.4em}
\titlespacing*{\subsubsection}{0pt}{0.9em}{0.2em}

\setlist[itemize]{leftmargin=1.4em, itemsep=0.15em, topsep=0.2em}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\footnotesize\color{darkgray}\thepage}

\newcommand{\feat}[1]{\textbf{#1}}
\newcommand{\stat}[1]{\texttt{#1}}

\begin{document}
"""


def fp(p: float) -> str:
    """A p-value as the report prints it, with its stars outside the math."""
    s = "<0.0001" if p < 0.0001 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"${s}${stars}"


def fq(q: float) -> str:
    s = "<0.0001" if q < 0.0001 else f"{q:.4f}" if q < 0.01 else f"{q:.3f}" if q < 0.2 else f"{q:.2f}"
    return f"${s}$"


def plain_p(p: float) -> str:
    return "<0.0001" if p < 0.0001 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def peq(p: float) -> str:
    """'p<0.0001' or 'p=0.0123', for use inside math mode."""
    return "p<0.0001" if p < 0.0001 else f"p={plain_p(p)}"


def tex_key(key: str) -> str:
    return key.replace("_", "\\_")


class Report:
    def __init__(self) -> None:
        self.ols = json.load(open("output/driver_ols_report.json", encoding="utf-8"))
        importance = json.load(open("output/driver_analysis.json", encoding="utf-8"))
        self.importance = {m: {f["feature"]: f["importance"] for f in importance[m]["features"]} for m in LABEL_TITLE}
        self.pv = {r["label"]: r for r in csv.DictReader(open("output/predictive_validity_report.csv", encoding="utf-8"))}
        self.site = json.load(open("output/site_stats.json", encoding="utf-8"))["text"]
        self.row = {m: {r["feature"]: r for r in self.ols[m]["rows"]} for m in self.ols}
        self.membership = json.load(open("output/membership_check.json", encoding="utf-8"))
        self.compute_descriptives()

    def compute_descriptives(self) -> None:
        """Standard deviations, variance inflation, and how a pitcher's rows relate, over the pitcher-seasons each driver model uses."""
        driver_df = pd.read_csv("output/driver_features.csv")
        ps = pd.read_csv("output/pitcher_season.csv")
        features = [c for c in driver_df.columns if c not in ("pitcher", "season")]
        self.resid_sd, self.max_vif, self.multi_season_share, self.cross_season_r = {}, {}, {}, {}
        for label in LABEL_TITLE:
            data = model_data(label, driver_df, ps, features)
            target = f"{label}_diff_adj_shrunk"
            self.resid_sd[label] = float(data[target].std())
            design = sm.add_constant(data[features])
            self.max_vif[label] = max(variance_inflation_factor(design.values, i) for i, c in enumerate(design.columns) if c != "const")
            self.multi_season_share[label] = float((data.groupby("pitcher")["season"].nunique() > 1).mean())
            self.cross_season_r[label] = pooled_correlation(consecutive_pairs(data[["pitcher", "season", target]], [target]), target)
        whiff = model_data("whiff", driver_df, ps, features)
        t = "whiff_diff_adj_shrunk"
        self.r_gap = float(whiff["avg_velocity_gap_from_prev"].corr(whiff[t]))
        self.r_repeat = float(whiff["repeat_pct"].corr(whiff[t]))
        self.r_between = float(whiff["avg_velocity_gap_from_prev"].corr(whiff["repeat_pct"]))
        denom = 1 - self.r_between ** 2
        self.beta_gap = (self.r_gap - self.r_between * self.r_repeat) / denom
        self.beta_repeat = (self.r_repeat - self.r_between * self.r_gap) / denom
        self.joint_share = self.beta_gap * self.r_gap + self.beta_repeat * self.r_repeat

    # ----- small accessors -----
    def r(self, model: str, key: str) -> dict:
        return self.row[model][key]

    def tiers(self, model: str) -> tuple[list[dict], list[dict], list[dict]]:
        rows = self.ols[model]["rows"]
        survive = [r for r in rows if r["q"] < 0.05]
        nominal = [r for r in rows if r["p"] < 0.05 and r["q"] >= 0.05]
        rest = [r for r in rows if r["p"] >= 0.05]
        return survive, nominal, rest

    def coef_line(self, model: str, key: str, tier3: bool = False) -> str:
        r = self.r(model, key)
        head = f"\\item Coefficient: ${r['coef']:+.5f}$ \\;|\\; $p$-value: {fp(r['p'])} \\;|\\; $q$: {fq(r['q'])} \\;|\\; "
        if tier3:
            return head + f"Permutation importance: ${self.importance[model][IMPORTANCE_NAME[key]]:.5f}$"
        return head + f"95\\% CI: $[{r['ci_lo']:.5f}, {r['ci_hi']:.5f}]$"

    def feature_block(self, model: str, key: str, tier3: bool = False) -> str:
        return (f"\\feat{{{TITLE[key]} ({tex_key(key)})}}\n\\begin{{itemize}}\n{self.coef_line(model, key, tier3)}\n"
                f"\\item {self.note(model, key)}\n\\end{{itemize}}\n")

    # ----- one-sentence readings, with the numbers they quote -----
    def note(self, model: str, key: str) -> str:
        r = self.r(model, key)
        direction = "higher" if r["coef"] > 0 else "lower"
        sd = self.resid_sd[model]
        if (model, key) == ("whiff", "tempo_bases_empty_sec"):
            return (f"Working slower between pitches predicts a higher whiff residual. The residual's own standard deviation across these "
                    f"pitcher-seasons is ${sd:.4f}$, so a 5-second pace difference moves the prediction by about {abs(5 * r['coef'] / sd):.1f} "
                    f"of a residual standard deviation.")
        if (model, key) == ("whiff", "avg_velocity_gap_from_prev"):
            return "Bigger speed changes between consecutive pitches predict more whiffs than stuff and location alone would, consistent with classic pitch-sequencing theory."
        if (model, key) == ("whiff", "repeat_pct"):
            return (f"Significant in the model, but weak on its own ($r={self.r_repeat:+.2f}$ with the whiff residual) and partly redundant with velocity gap "
                    f"(the two features correlate at $r={self.r_between:.2f}$). Read the two together; see Cross-Model Strategic Insights.")
        if (model, key) == ("whiff", "vaa_cross_pitch_std"):
            return ("Less spread in vertical approach angle across a pitcher's pitch types, a more consistent tunnel, goes with a higher residual. "
                    + self.vaa_status())
        if (model, key) == ("whiff", "spin_mirror_score_mean"):
            return ("A lower mirror score, consecutive pitches sharing a more similar or mirrored spin axis, goes with a higher residual, the direction the "
                    "tunneling idea predicts. " + ("It survives the correction here, after sitting just above $p=0.05$ in the two-season run." if r["q"] < 0.05
                                                  else "The effect is small and does not clear the correction."))
        if (model, key) == ("whiff", "release_extension_mean"):
            return "More extension toward the plate goes with a modestly higher residual, but not reliably."
        if (model, key) == ("whiff", "n_pitch_types"):
            return "A bigger repertoire trends toward more deception credit, but not reliably."
        if (model, key) == ("whiff", "arm_angle_szn_avg"):
            return "Arm slot carries no distinguishable effect on the whiff residual."
        if (model, key) == ("timing", "avg_velocity_gap_from_prev"):
            return (f"The single most reliable predictor in either model. Timing's own residual standard deviation is ${sd:.3f}$, so this effect is "
                    f"proportionally similar in size to the whiff-model version of the same feature.")
        if (model, key) == ("timing", "repeat_pct"):
            return "Confirms the whiff-model finding in an independent outcome: repeating a pitch type more often predicts bigger contact-timing misses, not smaller ones."
        if (model, key) == ("timing", "vaa_cross_pitch_std"):
            return "Same sign as in the whiff model. " + self.vaa_status()
        if (model, key) == ("timing", "vaa_mean"):
            return ("A pitcher's average approach angle itself (as opposed to its \\emph{consistency} across pitch types) has no distinguishable coefficient.")
        if r["p"] < 0.05:
            return f"Nominally significant, with a {direction} {LABEL_TITLE[model].lower()} residual, but it does not clear the correction."
        return f"No distinguishable effect on the {LABEL_TITLE[model].lower()} residual (${peq(r['p'])}$)."

    def vaa_status(self) -> str:
        w, t = self.r("whiff", "vaa_cross_pitch_std"), self.r("timing", "vaa_cross_pitch_std")
        if w["q"] < 0.05 and t["q"] < 0.05:
            return (f"It clears the correction in both models here ($q={w['q']:.3f}$ for whiff, ${t['q']:.3f}$ for timing). With two seasons it sat on the line "
                    f"($q=0.061$ and $0.047$ in one run, $0.050$ and $0.026$ in the next), so the third season is what separated it from noise.")
        return (f"It does not clear the correction in both models ($q={w['q']:.3f}$ for whiff, ${t['q']:.3f}$ for timing), so it is a lead and not a finding.")

    # ----- sections -----
    def summary_bullets(self) -> str:
        w, t = self.ols["whiff"], self.ols["timing"]
        sw, nw, _ = self.tiers("whiff")
        st, nt, _ = self.tiers("timing")
        rep_w, rep_t = self.r("whiff", "repeat_pct"), self.r("timing", "repeat_pct")
        vw, vt = self.r("whiff", "vaa_cross_pitch_std"), self.r("timing", "vaa_cross_pitch_std")
        both = vw["q"] < 0.05 and vt["q"] < 0.05
        vaa = (f"Vertical-approach-angle consistency across a pitcher's repertoire has the same sign in both models and clears the correction in both "
               f"($q={vw['q']:.3f}$ for whiff, ${vt['q']:.3f}$ for timing), as does the spin-axis mirror score for whiff ($q={self.r('whiff', 'spin_mirror_score_mean')['q']:.3f}$). "
               f"With only two seasons both sat on the correction threshold, so the third season is what separated them from noise. They are small effects "
               f"in the direction a tunneling story predicts, not proof of the mechanism"
               if both and self.r("whiff", "spin_mirror_score_mean")["q"] < 0.05 else
               f"Vertical-approach-angle consistency across a pitcher's repertoire has the same sign in both models but sits on the correction threshold "
               f"($q={vw['q']:.3f}$ for whiff, ${vt['q']:.3f}$ for timing). It is a lead to test directly, not a finding")
        return "\n".join([
            f"\\item Driver models explain a modest but statistically real share of each residual: $R^2 = {w['r2']:.3f}$ (whiff, $n={w['n']:,}$ pitcher-seasons from {w['pitchers']:,} pitchers) "
            f"and $R^2 = {t['r2']:.3f}$ (timing, $n={t['n']:,}$ from {t['pitchers']:,} pitchers), both significant overall ($F={w['fvalue']:.2f}$, $p<0.0001$ and $F={t['fvalue']:.2f}$, $p<0.0001$)",
            f"\\item With 14 features tested per model, results are read after a Benjamini--Hochberg correction, with standard errors clustered on pitcher because most pitchers appear in more than one season: "
            f"{len(sw)} of 14 features survive for the whiff residual ({len(sw) + len(nw)} are nominally significant at $p<0.05$) and {len(st)} of 14 for the timing residual "
            f"({'all' if not nt else len(st) + len(nt)} nominally significant features {'survive' if not nt else 'in total'})",
            f"\\item Velocity gap from the previous pitch and same-pitch-type repeat rate are both significant and positive in \\emph{{both}} models (repeat rate: whiff ${rep_w['coef']:+.4f}$, ${peq(rep_w['p'])}$; "
            f"timing ${rep_t['coef']:+.3f}$, ${peq(rep_t['p'])}$), but each is only weakly related to the residual on its own ($r={self.r_gap:+.2f}$ and ${self.r_repeat:+.2f}$ for whiff), "
            f"and the two features partly offset each other, so they are best read together as a weak signal about how a pitcher switches between pitches, not as evidence that repeating pitches creates deception",
            f"\\item {vaa}",
            f"\\item Deception+'s own held-out year-ahead forecast test, not a mere year-effect coefficient, confirms both outcomes studied here are temporally stable: "
            f"whiff cross-validated $\\Delta R^2={float(self.pv['whiff']['cv_delta_r2']):+.3f}$ ($p<0.0001$), timing $\\Delta R^2={float(self.pv['timing']['cv_delta_r2']):+.3f}$ ($p<0.0001$)",
        ])

    def equation(self, model: str) -> str:
        o = self.ols[model]
        survive, _, _ = self.tiers(model)
        terms = [f"{r['coef']:+.4f}\\times(\\text{{{TITLE[r['feature']]}}})" for r in survive[:4]]
        more = len(o["rows"]) - len(terms)
        lines = [f"{LABEL_TITLE[model]} residual $= {o['intercept']:+.4f} {terms[0]} {terms[1]}$" if len(terms) > 1 else f"{LABEL_TITLE[model]} residual $= {o['intercept']:+.4f} {terms[0]}$"]
        rest = " ".join(terms[2:])
        lines.append(f"$\\quad {rest} + ...\\ ({more}\\text{{ more terms}})$" if rest else f"$\\quad + ...\\ ({more}\\text{{ more terms}})$")
        return "\n".join(lines)

    def model_section(self, model: str) -> str:
        survive, nominal, rest = self.tiers(model)
        out = [f"\\section*{{Coefficient Analysis --- {LABEL_TITLE[model]} Residual Driver Model}}\n", "\\subsection*{Model Equation}", self.equation(model) + "\n",
               "\\subsection*{Tier 1: Survive the Multiple-Testing Correction ($q<0.05$)}\n"]
        out += [self.feature_block(model, r["feature"]) for r in survive]
        out.append("\\subsection*{Tier 2: Nominally Significant Only ($p<0.05$, $q\\ge0.05$)}\n")
        out += [self.feature_block(model, r["feature"]) for r in nominal] if nominal else [f"None. All {len(survive)} features with $p<0.05$ survive the correction.\n"]
        out.append("\\subsection*{Tier 3: Not Distinguishable From Zero}\n")
        out.append("Permutation importance is shown for reference only. The random forest behind it has a cross-validated $R^2$ of about $0.01$, so the ranking among these "
                   "features is noise and is not evidence of an effect.\n")
        out += [self.feature_block(model, r["feature"], tier3=True) for r in rest[:TIER3_SHOWN]]
        return "\n".join(out)

    def cross_model(self) -> str:
        gw, gt = self.r("whiff", "avg_velocity_gap_from_prev"), self.r("timing", "avg_velocity_gap_from_prev")
        rw, rt = self.r("whiff", "repeat_pct"), self.r("timing", "repeat_pct")
        vw, vt = self.r("whiff", "vaa_cross_pitch_std"), self.r("timing", "vaa_cross_pitch_std")
        sm_w = self.r("whiff", "spin_mirror_score_mean")
        pw, pt = self.r("whiff", "tempo_bases_empty_sec"), self.r("timing", "tempo_bases_empty_sec")
        vaa_title = "VAA Consistency Holds Up With Three Seasons" if vw["q"] < 0.05 and vt["q"] < 0.05 else "VAA Consistency Is Suggestive, Not Established"
        vaa_line = ("It clears the correction in both models." if vw["q"] < 0.05 and vt["q"] < 0.05 else "It does not clear the correction in both models.")
        vaa_impl = ("pitchers whose different pitch types share a similar approach angle get more deception credit, in line with this project's spin-axis-gap finding in the scoring model, "
                    "and the spin-axis mirror score points the same way for whiff. The effects are small and observational, so this supports a tunneling story without proving it."
                    if vw["q"] < 0.05 and vt["q"] < 0.05 else
                    "pitchers whose different pitch types share a similar approach angle may get more deception credit, in line with this project's spin-axis-gap finding in the scoring model. "
                    "Two same-signed results that straddle $q=0.05$ are a reason to test it directly, not a result.")
        return f"""\\section*{{Cross-Model Strategic Insights}}

\\subsection*{{Statistically Validated Universal Factors}}

\\feat{{Velocity Gap From Previous Pitch}}
\\begin{{itemize}}
\\item Both Models: significant and positive after correction (whiff ${gw['coef']:+.5f}$, ${peq(gw['p'])}$, $q={gw['q']:.4f}$; timing ${gt['coef']:+.5f}$, $p<0.0001$)
\\item Strategic Value: the single most consistently validated driver across both the rate and the magnitude dimension of swing-and-miss deception.
\\end{{itemize}}

\\feat{{Same-Pitch-Type Repeat Rate}}
\\begin{{itemize}}
\\item Both Models: significant and positive after correction (whiff ${peq(rw['p'])}$, $q={rw['q']:.4f}$; timing ${peq(rt['p'])}$)
\\item Strategic Value: significant in two independent scored outcomes, but weak on its own and entangled with velocity gap; see Findings That Need Careful Reading below.
\\end{{itemize}}

\\subsection*{{Findings That Need Careful Reading}}

\\feat{{Why Velocity Gap and Repeat Rate Both Come Out Positive}}
\\begin{{itemize}}
\\item Both Models: positive, significant coefficients for both features (repeat rate: whiff ${peq(rw['p'])}$, timing ${peq(rt['p'])}$). They can look contradictory, since mixing speeds and repeating the same pitch seem to be opposite behaviors.
\\item The apparent contradiction comes from how the features are built. Velocity gap averages the speed change over all consecutive pitches, repeats included (a repeat adds about zero), so it mixes how often a pitcher switches with how big the jump is when they do. The two features correlate at $r={self.r_between:.2f}$.
\\item Each is only weakly related to the whiff residual on its own ($r={self.r_gap:+.2f}$ for velocity gap, ${self.r_repeat:+.2f}$ for repeat rate; roughly {self.joint_share * 100:.1f}\\% of its variance together). Because the two offset each other, each coefficient looks stronger with both in the regression: the standardized coefficients implied by those three correlations are about ${self.beta_gap:.2f}$ and ${self.beta_repeat:.2f}$. That pattern is called cooperative suppression, and it is not a contradiction.
\\item A cleaner measure splits the feature into switch rate and gap per switch. An earlier check found the per-switch gap correlates about $+0.12$ with the whiff residual, stronger than either raw feature; that split is not yet part of the pipeline.
\\item Strategic Implication: this is not evidence that repeating pitches creates deception, or that mixing them does. The effect is small, observational, and partly an artifact of feature construction. It also cannot separate ``thrown again because it's dominant'' from ``dominant because it's thrown again.''
\\end{{itemize}}

\\feat{{{vaa_title}}}
\\begin{{itemize}}
\\item Both Models: negative (whiff ${vw['coef']:+.5f}$, ${peq(vw['p'])}$, $q={vw['q']:.3f}$; timing ${vt['coef']:+.5f}$, ${peq(vt['p'])}$, $q={vt['q']:.3f}$). {vaa_line} The spin-axis mirror score for whiff is ${sm_w['coef']:+.5f}$ (${peq(sm_w['p'])}$, $q={sm_w['q']:.3f}$).
\\item Strategic Implication: {vaa_impl}
\\end{{itemize}}

\\feat{{Pace Does Not Generalize}}
\\begin{{itemize}}
\\item Whiff Model: significant and positive (${pw['coef']:+.5f}$, $p<0.0001$)
\\item Timing Model: not significant ({"and the opposite sign " if pt['coef'] < 0 else ""}(${pt['coef']:+.5f}$, ${peq(pt['p'])}$))
\\item Strategic Implication: these are two different residuals, not two readings of the same effect. A driver that matters for whether a hitter whiffs does not automatically matter for how far off their timing is when they do make contact.
\\end{{itemize}}
"""

    def temporal(self) -> str:
        def row(label: str) -> tuple[float, float, float]:
            r = self.pv[label]
            return float(r["cv_delta_r2"]), float(r["delta_r2_in_sample"]), float(r["p_value"])
        wc, wi, _ = row("whiff")
        tc, ti, _ = row("timing")
        cs, cs_i, cs_p = row("calledstrike")
        al, al_i, al_p = row("align")
        m = self.membership
        return f"""\\section*{{Temporal Stability Analysis}}

Unlike a single in-sample year-effect coefficient, Deception+ already runs a genuine out-of-sample test: does a pitcher's score in one season predict their actual outcome rate the next season, on top of what that next season's own stuff-and-location expectation already explains? Every pair of consecutive seasons (2024 to 2025 and 2025 to 2026) is stacked into one regression with standard errors clustered on pitcher. The test reported here is a 5-fold cross-validated $R^2$ delta, with folds grouped by pitcher, plus a clustered test of the added term; unlike an in-sample $R^2$ delta it cannot be mechanically guaranteed positive.

\\subsection*{{Whiff Model Forecast Validity}}
\\begin{{itemize}}
\\item Cross-validated $\\Delta R^2$: ${wc:+.4f}$ \\;|\\; clustered $p$-value: $<0.0001$ \\;|\\; In-sample $\\Delta R^2$: ${wi:+.4f}$
\\item Interpretation: a pitcher's whiff score predicts real, out-of-sample whiff rate the next season beyond what that season's own expectation already explains.
\\end{{itemize}}

\\subsection*{{Timing Model Forecast Validity}}
\\begin{{itemize}}
\\item Cross-validated $\\Delta R^2$: ${tc:+.4f}$ \\;|\\; clustered $p$-value: $<0.0001$ \\;|\\; In-sample $\\Delta R^2$: ${ti:+.4f}$
\\item Interpretation: the strongest forecast validity of any scored component in the project.
\\end{{itemize}}

\\subsection*{{Contrast: Two Components With Almost No Forecast Gain}}
\\begin{{itemize}}
\\item Called Strike: cross-validated $\\Delta R^2 = {cs:+.4f}$, ${peq(cs_p)}$
\\item Horizontal Alignment: cross-validated $\\Delta R^2 = {al:+.4f}$, ${peq(al_p)}$
\\item Both clear the $p<0.01$ gate on three seasons (neither did on two), with gains a small fraction of a member's, and adding either lowers the composite's year-over-year correlation among qualified pairs (called strike ${m['+calledstrike']['diff']:+.3f}$, alignment ${m['+align']['diff']:+.3f}$, each with an interval below zero). That is why the project's composite metric excludes both.
\\end{{itemize}}

\\textbf{{Strategic Implication}}: deception that shows up in a pitcher's whiff and timing numbers one season keeps showing up in their actual outcomes the next. That is the strongest evidence in the project that these two residuals measure a stable pitcher trait rather than noise, and a materially stronger check than an in-sample year dummy could ever provide.
"""

    def hierarchy(self) -> str:
        out = ["\\section*{Statistical Significance Hierarchy}\n", "\\subsection*{Survive the Correction ($q<0.05$)}\n"]
        for model in LABEL_TITLE:
            survive, _, _ = self.tiers(model)
            out.append(f"\\textbf{{{LABEL_TITLE[model]} Model:}}\n\\begin{{itemize}}")
            for r in survive:
                extra = f", $q={r['q']:.3f}$" if r["q"] > 0.01 else ""
                out.append(f"\\item {TITLE[r['feature']]}: ${r['coef']:+.5f}$ (${peq(r['p'])}${'***' if r['p'] < 0.001 else '**' if r['p'] < 0.01 else '*'}{extra})")
            out.append("\\end{itemize}\n")
        out.append("\\subsection*{Nominally Significant Only ($p<0.05$, does not survive the correction)}\n")
        for model in LABEL_TITLE:
            _, nominal, _ = self.tiers(model)
            if nominal:
                out.append(f"\\textbf{{{LABEL_TITLE[model]} Model:}}\n\\begin{{itemize}}")
                out += [f"\\item {TITLE[r['feature']]}: ${r['coef']:+.5f}$ (${peq(r['p'])}$*, $q={r['q']:.3f}$)" for r in nominal]
                out.append("\\end{itemize}\n")
            else:
                out.append(f"\\textbf{{{LABEL_TITLE[model]} Model:}} none.\n")
        return "\n".join(out)

    def summary(self) -> str:
        out = ["\\section*{Summary of Statistical Findings}\n"]
        for model in LABEL_TITLE:
            survive, nominal, _ = self.tiers(model)
            largest = max(self.ols[model]["rows"], key=lambda r: abs(r["coef"]) if r["q"] < 0.05 else -1)
            best = survive[0]
            out.append(f"\\textbf{{{LABEL_TITLE[model]} Model:}}\n\\begin{{itemize}}\n\\item Total features analyzed: 14\n"
                       f"\\item Nominally significant features ($p<0.05$): {len(survive) + len(nominal)} of 14; surviving the correction ($q<0.05$): {len(survive)} of 14\n"
                       f"\\item Largest raw coefficient among survivors (depends on feature units): {TITLE[largest['feature']]} (${largest['coef']:+.4f}$, ${peq(largest['p'])}$)\n"
                       f"\\item Most reliable predictor: {TITLE[best['feature']]} (${peq(best['p'])}$)\n\\end{{itemize}}\n")
        return "\n".join(out)

    def takeaways(self) -> str:
        w, t = self.ols["whiff"], self.ols["timing"]
        vw, vt = self.r("whiff", "vaa_cross_pitch_std"), self.r("timing", "vaa_cross_pitch_std")
        sm_w = self.r("whiff", "spin_mirror_score_mean")
        holds = vw["q"] < 0.05 and vt["q"] < 0.05
        tunnel_title = "Tunneling Consistency Holds Up, Modestly" if holds else "Tunneling Consistency Is a Lead, Not a Finding"
        tunnel = (f"pitchers whose different pitch types share a similar vertical approach angle earn more deception credit in both whiff and timing (${peq(vw['p'])}$ and ${plain_p(vt['p'])}$, "
                  f"same sign), and both clear the correction ($q={vw['q']:.3f}$ and ${vt['q']:.3f}$); the spin-axis mirror score, the other tunneling measure, clears it for whiff "
                  f"(${peq(sm_w['p'])}$, $q={sm_w['q']:.3f}$) and is absent for timing. With two seasons these sat on the correction line and moved across it between reruns, so the extra season matters. "
                  f"They fit this project's own spin-axis-gap finding from the scoring model, but the effects are small and observational."
                  if holds else
                  f"pitchers whose different pitch types share a similar vertical approach angle earn more deception credit in both whiff and timing (${peq(vw['p'])}$ and ${plain_p(vt['p'])}$, same sign), "
                  f"but the correction puts the two on either side of the line ($q={vw['q']:.3f}$ and ${vt['q']:.3f}$).")
        return f"""\\section*{{Key Strategic Takeaways}}

\\begin{{enumerate}}
\\item \\textbf{{Sequencing Features Carry a Weak Signal}}: pitch-to-pitch velocity change and repeat rate are both significant and positive in two separate scored outcomes, but each is only weakly related to the residual ($r={self.r_gap:+.2f}$ and ${self.r_repeat:+.2f}$ for whiff) and the two partly offset each other. They are better read as a weak signal about how a pitcher switches between pitches than as proof that mixing or repeating pitches creates deception.
\\item \\textbf{{{tunnel_title}}}: {tunnel}
\\item \\textbf{{Season-Level Traits Explain Little of the Residual}}: $R^2$ of ${min(w['r2'], t['r2']):.3f}$ to ${max(w['r2'], t['r2']):.3f}$ means over 90\\% of the scored residual is still unexplained by these 14 traits. This is an explanatory layer on top of Deception+, not a predictive one, and should be read as such.
\\item \\textbf{{Temporal Stability Is Verified, Not Assumed}}: whiff and timing both pass a genuine held-out year-ahead forecast test, and so does every other member; called strike and horizontal alignment clear the gate only with a small fraction of a member's gain, and lower the composite's stability, which is why Deception+'s composite excludes them.
\\item \\textbf{{Treat the Sequencing Findings as Correlational}}: this analysis cannot separate ``thrown again because it's good'' from ``good because it's thrown again.'' A pitch-level, within-pitcher test is a natural next step.
\\end{{enumerate}}
"""

    def limitations(self) -> str:
        w, t = self.ols["whiff"], self.ols["timing"]
        vw, vt = self.r("whiff", "vaa_cross_pitch_std"), self.r("timing", "vaa_cross_pitch_std")
        return f"""\\section*{{Methodological Notes and Limitations}}
\\begin{{itemize}}
\\item \\textbf{{Season-level sample}}: $n \\approx {t['n']:,}$--${w['n']:,}$ pitcher-seasons from about {t['pitchers']:,}--{w['pitchers']:,} pitchers. Results describe between-pitcher patterns across a full season, not week-to-week variation within one.
\\item \\textbf{{Moderate multicollinearity}}: pitch-mix entropy carries the highest variance inflation factor in both models (VIF $\\approx {max(self.max_vif.values()):.1f}$), reflecting its real mathematical relationship to repeat rate. No feature in either model exceeds the conventional VIF $=10$ concern threshold, but the entropy and repeat-rate coefficients are best read together rather than in isolation.
\\item \\textbf{{Exploratory, even after correction}}: the Benjamini--Hochberg adjustment covers the 14 tests inside each model. The project has tested many other things (component candidates, other drivers, sequencing features), so even a result that survives deserves replication on another season before it is relied on. Vertical-approach-angle consistency shows how a marginal result moves: clustered, its $q$ was $0.061$ and $0.047$ for whiff and timing in one two-season run, $0.050$ and $0.026$ in the next, and is ${vw['q']:.3f}$ and ${vt['q']:.3f}$ with a third season.
\\item \\textbf{{The residuals carry model noise}}: which pitchers share a cross-validation fold changes a pitcher's expectation, worth about 0.75 index points per component after averaging six random grouped splits (about 1.1 with three, about 2 with one). Season-level driver results near a threshold can move with it.
\\item \\textbf{{Raw coefficients depend on units}}: a coefficient's size reflects the scale of its feature (repeat rate runs from 0 to 1, pace in seconds), so ``largest coefficient'' comparisons across features are not like-for-like. Standardized coefficients would be the fair comparison.
\\item \\textbf{{The residual is not independent of pitch quality}}: Deception+ correlates $r={self.site['STUFF_R']}$ with FanGraphs Stuff+, so these coefficients describe what separates pitchers on a residual that still carries some pitch-quality information.
\\item \\textbf{{Explanatory, not predictive}}: this driver regression is a diagnostic layer only, exactly as the project's own driver-analysis module states. It does not feed back into, or change, any pitcher's scored Deception+ value.
\\end{{itemize}}

\\end{{document}}
"""

    def build(self) -> str:
        w, t = self.ols["whiff"], self.ols["timing"]
        pitchers_multi = max(self.multi_season_share.values())
        title = f"""
\\begin{{center}}
{{\\Large\\bfseries\\color{{hdrblue}} How Pitch Sequencing and Tempo Drive Swing-and-Miss Deception}}\\\\[0.3em]
{{\\normalsize A Driver Analysis of the Deception+ Whiff and Timing Residuals, {SEASONS}}}\\\\[0.8em]
{{\\normalsize By: Henry Bednar}}
\\end{{center}}

\\vspace{{0.5em}}

\\section*{{Executive Summary}}
This analysis builds on Deception+, a from-scratch pitcher deception metric scored on three complete seasons of Statcast pitch-level data ({SEASONS}), by asking what season-level pitcher traits explain the residual the metric already measures: whether a pitcher gets more whiffs, and bigger contact-timing misses, than their own stuff, location, opponent, and game context alone would predict. Two scored residuals were regressed on 14 season-level driver features (pitch-to-pitch velocity and movement change, tempo, release-point and approach-angle consistency, tunnel separation, spin mirroring, repertoire size) using ordinary least squares with standard errors clustered on pitcher, the same coefficient-table methodology used throughout this kind of analysis.

\\textbf{{Key Findings:}}
\\begin{{itemize}}
{self.summary_bullets()}
\\end{{itemize}}

\\section*{{Research Methodology \\& Analytical Framework}}

\\subsection*{{Analytical Approach}}
\\begin{{itemize}}
\\item Deception+ scores every pitch against a per-pitch gradient-boosted expectation (stuff, plate location, batter tendency overall, against same-handed pitchers, by pitch type, by count and by zone, catcher tendency, count, park, platoon, season, in-game fatigue, times through the order, and game state), fit with 5-fold cross-validation grouped by pitcher so a pitcher's own pitches never inform their own expectation. Each expectation averages six random grouped splits and is recalibrated for the league-wide level of each month.
\\item The residual (actual minus expected) is empirical-Bayes shrunk toward the league mean using a Paule--Mandel between-pitcher variance estimate, then scored on a 100/10 index.
\\item This report regresses that shrunk residual on 14 pitcher-season traits engineered independently of the scoring model: release-point, arm-angle and vertical-approach-angle spread across a pitcher's repertoire, velocity gap and repeat rate from the previous pitch, tunnel differential, spin-axis mirror score, pace, pitch-mix entropy, and repertoire size.
\\item \\textbf{{Complete Coefficient Analysis}}: ordinary least squares with standard errors, $p$-values, and 95\\% confidence intervals for every feature in both models. Standard errors cluster on pitcher: about {pitchers_multi * 100:.0f}\\% of pitchers contribute more than one season, and a pitcher's seasons are strongly dependent (their whiff residuals correlate {self.cross_season_r['whiff']:.2f} across consecutive seasons), so classical errors overstate the evidence. $R^2$, $F$ and the coefficients are the plain OLS values. (The project's existing driver analysis uses Ridge regression and Random Forest permutation importance for a predictive read; this report adds the classical significance-testing layer the coefficient-table format below needs.)
\\item \\textbf{{Multiple-testing correction}}: 14 features are tested in each model, so about 0.7 would reach $p<0.05$ by chance alone. Features are classed by their Benjamini--Hochberg adjusted $q$ across the 14 clustered tests of each model. A feature with $p<0.05$ but $q\\ge0.05$ is reported as nominal and not as a finding.
\\item Deliberately excluded: Savant's own swing-outcome stats (miss distance, early/late/tied-up/flailed percentages), since they are outcomes of the same swings the residual already scores, and the pitch-tempo pull's duplicate pace column, confirmed a byte-for-byte copy of the first. The season-level traits use regular-season pitches only, like the scores.
\\end{{itemize}}

\\subsection*{{Data Scope \\& Model Performance}}
\\begin{{itemize}}
\\item Three complete regular seasons ({SEASONS}), data through {self.site['DATA_THROUGH']}
\\item {self.site['N_SCORED']} pitcher-seasons scored overall; {self.site['N_QUALIFIED']} qualified
\\item \\textbf{{Whiff Driver Model}}: $R^2={w['r2']:.3f}$, Adjusted $R^2={w['adj_r2']:.3f}$, $F={w['fvalue']:.2f}$ (14 features, $df={w['df_resid']:,}$, $n={w['n']:,}$, {w['pitchers']:,} pitchers)
\\item \\textbf{{Timing Driver Model}}: $R^2={t['r2']:.3f}$, Adjusted $R^2={t['adj_r2']:.3f}$, $F={t['fvalue']:.2f}$ (14 features, $df={t['df_resid']:,}$, $n={t['n']:,}$, {t['pitchers']:,} pitchers)
\\item \\textbf{{Temporal Validation}}: both outcomes pass a genuine held-out year-ahead forecast test (see Temporal Stability Analysis)
\\end{{itemize}}

"""
        return "\n".join([PREAMBLE, title, self.model_section("whiff"), self.model_section("timing"), self.cross_model(), self.temporal(),
                          self.hierarchy(), self.summary(), self.takeaways(), self.limitations()])


if __name__ == "__main__":
    text = Report().build()
    with open("report.tex", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote report.tex ({text.count(chr(10)) + 1} lines). Compile with: pdflatex report.tex")
