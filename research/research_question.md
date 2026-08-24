# Research Question

## Final Research Question

**What structural and semantic properties of questions predict when large language models will be confidently wrong?**

### Sub-questions

- **RQ1**: Which question-level features (difficulty, reasoning steps, misleading premises, numerical content, domain, ambiguity, option similarity) are significant predictors of confident failure?
- **RQ2**: Are the same features predictive across different LLMs, or do models have different failure profiles?
- **RQ3**: How much variance in per-item miscalibration can question-level features explain?

---

## Gap Justification

Existing research has established that LLMs are overconfident (Geng et al., 2024; Chhikara, 2025; Kumaran et al., 2026) and has proposed various methods to measure and improve calibration (Tian et al., 2023; Zhang et al., 2024; Xiao et al., 2025; ICLR, 2026). However, all prior work measures calibration at the aggregate dataset level. No study has examined what structural and semantic properties of individual questions predict when an LLM will be confidently wrong.

## Why This Question Matters

Understanding *what kinds of questions* cause confident failures has direct practical value. If question properties can predict confident failure, those properties can be computed *before* querying a model — enabling pre-screening, confidence adjustment, or human-in-the-loop routing for high-risk queries.

## Key Finding

**Answer option similarity** is the strongest predictor of confident failure (OR = 3.99, p < 0.0001). When answer choices are semantically similar to each other, models are nearly 4x more likely to be confidently wrong. This finding is novel — no prior paper in the calibration literature has measured or controlled for this variable.

## Candidate Questions Considered

1. **(Selected)** Question-level predictors of confident failures
2. Verbalized vs. logit-based confidence divergence
3. Calibration degradation across reasoning chain length
4. Distractor properties and calibration interaction
5. Cross-domain calibration transfer

Candidate 1 was selected for originality, feasibility, and practical significance. See `literature/research_gaps.md` for the full gap analysis.
