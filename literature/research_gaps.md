# Research Gaps Identified from Literature Review

Based on analysis of 9 papers on LLM confidence, calibration, and overconfidence.

---

## Gap A: No question-level predictors of confident failure (SELECTED)

All papers measure calibration at the aggregate dataset level (ECE across a dataset). No study has treated question-level features as independent variables to predict per-item miscalibration. Liu et al. (2025) explicitly note that standard metrics "fail to distinguish 'confidently wrong' responses from those with appropriate uncertainty."

**Status**: This gap was selected as our research question and tested experimentally.

---

## Gap B: Verbalized vs. logit-based confidence never compared head-to-head

Paper 02 (Tian et al.) argues verbalized > logit-based. Paper 08 (Kumaran et al.) uses logit-based exclusively. Paper 06 (Chhikara) uses verbalized exclusively. No paper extracts both on the same questions and compares them item-by-item.

## Gap C: Interaction between question structure and confidence method untested

It's possible that verbalized confidence is better calibrated for some question types but worse for others. No paper examines this.

## Gap D: Reasoning tasks almost entirely absent from calibration research

Only Paper 08 uses GSM-MC, and only as a transfer test. Multi-step reasoning, where errors can compound, is the least studied context despite being where confidence matters most.

## Gap E: Calibration vs. reasoning chain length never tested

Paper 05 (Liu et al.) identifies reasoning uncertainty as severely understudied (only 3 methods in their entire taxonomy). No paper tests whether longer CoT chains produce progressively worse calibration.

## Gap F: "Confidently wrong" phenomenon described but never dissected

Paper 05 notes standard metrics fail to distinguish confidently wrong responses. Paper 08 identifies choice-supportive bias as one mechanism. But no paper takes a set of confidently-wrong responses and asks: what made the model confident? What properties predicted this failure?

**This is essentially Gap A restated from a different angle.**

---

## Cross-Cutting Observations

| Theme | Status in Literature |
|-------|---------------------|
| Factual QA dominates | 7 of 7 empirical papers use factual QA |
| ECE is universal metric | All 9 papers use or discuss ECE |
| RLHF degrades calibration | Established by Papers 02, 03, 04, 06 |
| English-only | Universal limitation |
| One confidence method per study | No head-to-head comparisons |
| No manipulation of question properties | All papers take benchmarks as-is |

## Recurring Future Work Suggestions

- Extend to reasoning/math/code (Papers 02, 03, 05, 06)
- Extend to open-ended generation (Papers 01, 02, 03, 05, 09)
- Understand why RLHF hurts calibration (Papers 02, 04, 08)
- Multimodal calibration (Papers 01, 05)
- Better evaluation metrics beyond ECE (Papers 03, 05)
