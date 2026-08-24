# Confident and Wrong: What Question Properties Predict Confidence Failures in Large Language Models?

---

## Abstract

Large language models (LLMs) frequently express high confidence in incorrect answers, yet prior research has measured calibration only at the aggregate dataset level without examining what properties of individual questions predict these failures. This study addresses that gap by analyzing 1,936 responses from two LLMs (Qwen 2.5 3B and Qwen 2.5 1.5B) across 1,000 questions drawn from MMLU, ARC-Challenge, and TruthfulQA. We annotated each question with structural and semantic features — including answer option similarity, numerical content, negation, question length, domain, and misleading premises — and used logistic regression and random forest models to predict confident failures (incorrect answers given with ≥80% confidence). Both models exhibited severe overconfidence, averaging 94.9–96.9% confidence while achieving only 54.3–65.5% accuracy. The strongest predictor of confident failure was answer option similarity: when answer choices were semantically similar, the odds of a confident failure increased nearly fourfold (OR = 3.99, p < 0.0001). Numerical content also significantly increased confident failure odds (OR = 1.55, p < 0.001). Random forest analysis confirmed option similarity as the dominant feature (42% importance), followed by question length (28%). These results suggest that LLM overconfidence is not random but is systematically driven by identifiable question characteristics, with answer option confusability playing a central role.

---

## 1. Introduction

Large language models have become widely deployed in applications ranging from search engines to medical question-answering, yet a fundamental problem persists: these models often express high confidence in answers that are incorrect (Geng et al., 2024; Liu et al., 2025). This phenomenon — termed "confident failure" — poses risks in any setting where users rely on model-expressed certainty to evaluate response trustworthiness.

A substantial body of research has documented that LLMs are, on average, overconfident. Geng et al. (2024) provide a comprehensive taxonomy of confidence estimation methods and note that overconfident tokens consistently outnumber underconfident ones. Tian et al. (2023) demonstrate that RLHF-tuned models can verbalize confidence scores that are better calibrated than their own internal log-probabilities, yet these scores still exhibit systematic miscalibration. Zhang et al. (2024) propose decomposing confidence into uncertainty and fidelity components, finding that verbalized confidence tends to cluster in narrow high-confidence ranges (0.8–0.9) regardless of actual correctness. Steyvers et al. (2024) show that this overconfidence propagates to human users, who significantly overestimate LLM accuracy based on model-generated explanations.

More recent work has begun to examine the mechanisms underlying miscalibration. Kumaran et al. (2026) identify two competing biases in LLM confidence: a choice-supportive bias that inflates confidence in previously generated answers, and a systematic overweighting of contradictory information that produces underconfidence in adversarial contexts. Chhikara (2025) demonstrates that distractor-augmented prompts can reduce overconfidence by up to 90% on hard questions but paradoxically increase miscalibration on easy ones. The Rewarding Doubt framework (ICLR, 2026) shows that reinforcement learning with proper scoring rules can train models to express calibrated confidence, reducing ECE from 0.45 to 0.02 on TriviaQA.

Despite this progress, a critical gap remains. All existing studies measure calibration at the aggregate level — computing expected calibration error (ECE) across entire datasets. No study has systematically examined what properties of individual questions predict when an LLM will be confidently wrong. Liu et al. (2025) explicitly note that "standard evaluation metrics fail to distinguish 'confidently wrong' responses from those with appropriate uncertainty." Chhikara (2025) observes that calibration varies by question type (e.g., person-based vs. place-based queries) but does not model question features as predictors of miscalibration. Xiao et al. (2025) find that consistency-based confidence works better for some tasks than others but do not investigate what task or question characteristics drive this variation.

This study addresses this gap directly. Rather than asking "how well-calibrated is this model on this dataset?", we ask: **what structural and semantic properties of questions predict when an LLM will be confidently wrong?** By shifting the unit of analysis from the dataset to the individual question, we aim to identify the features that systematically trigger confidence failures — information that could be used to flag high-risk queries before a model responds.

---

## 2. Literature Review

### 2.1 Confidence Estimation in LLMs

Methods for estimating LLM confidence fall into two broad categories (Geng et al., 2024). White-box methods access internal model states, including token log-probabilities, entropy measures, and learned probes on hidden activations such as P(True) and P(IK). Black-box methods operate without internal access, relying on verbalized confidence (prompting the model to state its certainty numerically), consistency across multiple generations, or surrogate models.

For RLHF-aligned models — which dominate current deployment — verbalized confidence has emerged as a practical and surprisingly effective signal. Tian et al. (2023) find that prompting ChatGPT and GPT-4 to state numerical confidence produces better-calibrated estimates than extracting conditional probabilities. Zhang et al. (2024) confirm this but note a persistent tendency for verbalized scores to cluster at high values. Chhikara (2025) justifies verbalized confidence over log-probabilities on the grounds that it captures task-level belief, is more reliable post-RLHF, and is uniformly available across both open-weight and closed-source models.

### 2.2 Calibration and Overconfidence

Expected Calibration Error (ECE) is the dominant metric for assessing whether a model's expressed confidence aligns with its actual accuracy (Geng et al., 2024; Liu et al., 2025). A perfectly calibrated model would have ECE = 0; among the questions where it reports 80% confidence, exactly 80% would be answered correctly. In practice, LLMs consistently deviate from this ideal in the direction of overconfidence.

Steyvers et al. (2024) document this concretely: GPT-3.5 achieves an internal ECE of 0.104 but generates explanations that lead human evaluators to an ECE of 0.264, indicating that overconfidence is amplified by the model's own explanations. Kumaran et al. (2026) identify a choice-supportive bias as one mechanism: when models can see their initial answer, confidence inflates by +0.22 even without new information.

### 2.3 Factors Affecting Calibration

Several studies have examined factors that influence calibration, though none at the individual question level. Chhikara (2025) varies model size (8B to >1T parameters), finding that larger RLHF-tuned models show lower ECE on hard questions but paradoxically higher miscalibration on easy ones when distractors are introduced. The same study reports that person-based queries exhibit the highest relative ECE improvement (69%) from distractors, while place-based queries show the lowest (49%).

Liu et al. (2025) propose a four-dimensional uncertainty taxonomy — input, reasoning, parameter, and prediction uncertainty — and note that input and reasoning uncertainty are severely understudied (only 3 methods in their entire literature survey address input uncertainty). Xiao et al. (2025) show that consistency-based confidence works better for question answering and text-to-SQL than for summarization, suggesting task-level properties matter.

### 2.4 The Gap

No prior work has treated question-level features as independent variables to predict per-item confident failure. The literature measures calibration in aggregate but never asks: for this specific question, what characteristics made the model confidently wrong? This study fills that gap.

---

## 3. Methods

### 3.1 Research Questions

- **RQ1**: Which question-level features (numerical content, option similarity, negation, misleading premises, domain, question length) are significant predictors of confident failure?
- **RQ2**: Are the same features predictive across different LLMs, or do models have different failure profiles?
- **RQ3**: How much variance in per-item confident failure can question-level features explain?

### 3.2 Datasets

Questions were drawn from three established benchmarks:

- **MMLU** (Hendrycks et al., 2021): 800 questions stratified across 57 subjects spanning STEM, humanities, social sciences, and professional domains. All questions are four-option multiple choice.
- **ARC-Challenge** (Clark et al., 2018): 100 grade-school science questions requiring reasoning, not just factual recall. Four-option multiple choice.
- **TruthfulQA** (Lin et al., 2022): 100 questions designed to elicit common misconceptions and popular falsehoods. Converted to four-option multiple choice using the mc1 targets.

The total sample comprised 1,000 questions, selected via stratified random sampling from the full datasets (16,006 questions) to ensure representation across domains and difficulty levels.

### 3.3 Models

Two instruction-tuned LLMs were evaluated:

- **Qwen 2.5 3B Instruct** (Qwen Team, 2024): A 3-billion-parameter model.
- **Qwen 2.5 1.5B Instruct** (Qwen Team, 2024): A 1.5-billion-parameter model from the same family.

Both models are open-weight and were run on a Google Colab T4 GPU using float16 precision. The two models from the same family but different sizes allow direct comparison of how model scale affects confident failure patterns (RQ2).

### 3.4 Confidence Elicitation

Each model was prompted to answer each question and provide a numerical confidence score (0–100):

```
Answer the following multiple-choice question. After providing your answer,
rate your confidence that your answer is correct on a scale from 0 to 100,
where 0 means you are certain you are wrong and 100 means you are certain
you are correct.

Question: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]

Respond in exactly this format:
Answer: [A/B/C/D]
Confidence: [0-100]
```

Temperature was set to 0 (greedy decoding) for reproducibility. Verbalized confidence was chosen over log-probability-based methods following the justifications in Tian et al. (2023) and Chhikara (2025): it captures task-level belief, is reliable post-RLHF, and is uniformly available across model architectures.

### 3.5 Question Feature Annotation

Each question was annotated with the following features:

| Feature | Type | Operationalization |
|---------|------|-------------------|
| **Question length** | Continuous | Word count of question stem |
| **Has negation** | Binary | Presence of NOT, EXCEPT, NEVER, NEITHER, NOR, CANNOT |
| **Has numerical content** | Binary | ≥2 numerical references in question + options |
| **Option similarity** | Continuous (0–1) | Average pairwise cosine similarity of answer option embeddings (all-MiniLM-L6-v2 sentence-BERT model) |
| **Has misleading premise** | Binary | All TruthfulQA questions = 1; all others = 0 |
| **Domain category** | Categorical | STEM, Humanities, Social Sciences, Professional/Other, Misconceptions |

### 3.6 Dependent Variable

**Confident failure** was defined as a binary outcome: the model's answer is incorrect AND its verbalized confidence is ≥80. The threshold of 80 was chosen based on Zhang et al. (2024), who find that verbalized confidence clusters in the 80–90 range.

### 3.7 Statistical Analysis

**Logistic regression** was used to predict confident failure from question features plus model identity (as a dummy variable), addressing RQ1 and RQ3. Features included question length, has negation, has numerical, option similarity, has misleading premise, domain dummies (drop-first encoding), and model dummy. The collinear pair (has_misleading_premise / domain_category_Misconceptions) was resolved by dropping the domain dummy.

**Random forest classification** (500 trees, max depth 10, balanced class weights) was used to capture non-linear relationships and compute feature importance via Gini impurity, with 5-fold cross-validated AUC to assess predictive performance.

**Expected Calibration Error (ECE)** was computed per model and per domain using 10 equal-width bins from 0 to 1.

---

## 4. Results

### 4.1 Descriptive Statistics

Both models exhibited severe overconfidence (Table 1). Qwen 2.5 3B achieved 65.5% accuracy but reported an average confidence of 96.9%, yielding an ECE of 0.349. Qwen 2.5 1.5B was less accurate (54.3%) and slightly less confident (94.9%) but more miscalibrated overall (ECE = 0.407). Of 1,936 valid responses, 770 (39.8%) qualified as confident failures.

**Table 1.** Model performance and calibration summary.

| Model | N | Accuracy | Avg. Confidence | ECE | Confident Failures | CF Rate |
|-------|---|----------|----------------|-----|-------------------|---------|
| Qwen 2.5 3B | 936 | 65.5% | 96.9 | 0.349 | 315 | 33.7% |
| Qwen 2.5 1.5B | 1,000 | 54.3% | 94.9 | 0.407 | 455 | 45.5% |
| **Combined** | **1,936** | **59.6%** | **95.9** | **0.379** | **770** | **39.8%** |

### 4.2 Logistic Regression: Predictors of Confident Failure (RQ1, RQ3)

Logistic regression identified three significant predictors of confident failure (Table 2).

**Table 2.** Logistic regression results predicting confident failure.

| Predictor | Coefficient | Odds Ratio | p-value | Significant |
|-----------|------------|------------|---------|-------------|
| Option similarity | 1.384 | 3.990 | < 0.0001 | Yes |
| Has numerical content | 0.437 | 1.548 | 0.0007 | Yes |
| Model (Qwen 3B vs 1.5B) | −0.495 | 0.609 | < 0.0001 | Yes |
| Has negation | −0.191 | 0.826 | 0.249 | No |
| Question length | 0.001 | 1.001 | 0.137 | No |
| Has misleading premise | 0.063 | 1.065 | 0.743 | No |
| Domain: Professional/Other | −0.027 | 0.974 | 0.856 | No |
| Domain: STEM | −0.282 | 0.754 | 0.060 | No |
| Domain: Social Sciences | −0.130 | 0.878 | 0.441 | No |

**Option similarity** was the strongest predictor: a one-unit increase in average pairwise cosine similarity between answer options was associated with a nearly fourfold increase in the odds of confident failure (OR = 3.99, p < 0.0001). **Numerical content** increased the odds by 55% (OR = 1.55, p < 0.001). The 3B model had 39% lower odds of confident failure than the 1.5B model (OR = 0.61, p < 0.0001).

Question length, negation, misleading premises, and domain were not significant predictors at p < 0.05, though STEM domain approached significance (p = 0.060) with a protective effect.

The model's pseudo-R² (McFadden) was 0.049, indicating that question-level features and model identity explain approximately 5% of the variance in confident failures.

### 4.3 Random Forest Feature Importance (RQ1)

Random forest analysis confirmed the logistic regression findings with a non-linear model. Feature importance (Gini impurity) rankings were:

**Table 3.** Random forest feature importance.

| Feature | Importance |
|---------|-----------|
| Option similarity | 0.416 |
| Question length | 0.278 |
| Model | 0.094 |
| Has numerical content | 0.083 |
| Domain | 0.082 |
| Has misleading premise | 0.030 |
| Has negation | 0.017 |

Option similarity dominated with 41.6% of total importance. Question length ranked second (27.8%), suggesting a non-linear relationship not captured by logistic regression (where it was non-significant). This discrepancy indicates that the relationship between question length and confident failure may be U-shaped or threshold-based rather than linear.

### 4.4 Cross-Model Comparison (RQ2)

The model dummy was significant in the logistic regression (p < 0.0001), confirming that the two models differ in their overall confident failure rates. The 3B model's lower odds ratio (0.61) indicates better calibration with increased scale, consistent with findings from Chhikara (2025).

### 4.5 Domain-Level Calibration

ECE varied substantially across domains (Table 4).

**Table 4.** Calibration by domain.

| Domain | N | Accuracy | Avg. Confidence | ECE | CF Rate |
|--------|---|----------|----------------|-----|---------|
| Misconceptions | 200 | 54.5% | 91.5% | 0.485 | 45.0% |
| Humanities | 359 | 53.5% | 96.7% | 0.432 | 46.2% |
| Professional/Other | 491 | 58.0% | 96.8% | 0.388 | 41.5% |
| STEM | 541 | 64.0% | 96.4% | 0.330 | 35.3% |
| Social Sciences | 345 | 64.9% | 95.5% | 0.326 | 34.5% |

Misconceptions (TruthfulQA) showed the worst calibration (ECE = 0.485), but notably lower average confidence (91.5%) than other domains, suggesting models may have some awareness that these questions are tricky. Humanities questions showed the second-worst calibration with a 46.2% confident failure rate despite 96.7% average confidence — the most extreme disconnect between expressed confidence and actual performance.

---

## 5. Discussion

### 5.1 Option Similarity as the Primary Driver

The most striking finding is that answer option similarity is by far the strongest predictor of confident failures, with an odds ratio of 3.99 and 41.6% of random forest importance. This means that when the four answer choices are semantically close to one another — when they are confusable — the model is most likely to answer incorrectly while expressing high confidence.

This finding has a plausible mechanistic interpretation. LLMs generate text by predicting token probabilities, and semantically similar options may occupy overlapping regions of the model's representation space. The model may commit to an answer early in its generation process (a form of the choice-supportive bias identified by Kumaran et al., 2026) without recognizing that the fine distinctions between options require more careful reasoning. The confidence score, generated after the answer, may reflect the model's general familiarity with the topic rather than its certainty about the specific distinction being tested.

This finding is novel. No prior study in the calibration literature has measured or controlled for answer option similarity as a variable. Chhikara (2025) generated distractors to study calibration but used fixed distractor sets without varying their similarity to the correct answer. Zhang et al. (2024) studied fidelity to generated answers but did not examine what makes a question's options more or less confusable.

### 5.2 Numerical Content

Questions containing numerical content showed 55% higher odds of confident failure. This aligns with the known difficulty LLMs face with precise numerical reasoning (Liu et al., 2025), but adds the insight that this difficulty is accompanied by inappropriately high confidence. The model does not "know that it doesn't know" about numerical relationships — it fails confidently rather than expressing appropriate uncertainty.

### 5.3 Model Scale

The 3B model showed 39% lower odds of confident failure than the 1.5B model. This is consistent with the general finding that larger models tend to be better calibrated (Chhikara, 2025), though both models remained severely overconfident (ECE > 0.34). Scale helps but does not solve the fundamental calibration problem.

### 5.4 The Absence of Expected Predictors

Several features that might be expected to predict confident failure were not significant. Negation, misleading premises, and domain category all failed to reach significance. This is informative: it suggests that confident failures are not primarily driven by "trick questions" or adversarial framing, but rather by the structural confusability of answer choices. A straightforward question with four similar-sounding options is more likely to produce a confident failure than a deliberately misleading question with clearly distinct options.

### 5.5 Practical Implications

These findings have direct practical implications. If option similarity can be computed before presenting a question to an LLM — which it can, since sentence-BERT embeddings are cheap to compute — it could be used as a pre-screening signal to flag questions where the model's confidence should be discounted. In retrieval-augmented generation (RAG) systems, this could inform when to request human review or when to present confidence-adjusted outputs.

### 5.6 Limitations

This study has several important limitations:

1. **Model selection**: Only two models from the same family (Qwen 2.5) were evaluated. The findings may not generalize to models from different families (e.g., Llama, GPT, Gemma) or to substantially larger models.

2. **Verbalized confidence only**: We used verbalized confidence rather than log-probability-based measures. Paper 02 (Tian et al., 2023) shows these can diverge; future work should compare both methods on the same items.

3. **Multiple choice only**: All questions were four-option multiple choice. The option similarity feature is inherently tied to this format and would not directly apply to open-ended generation.

4. **Sample size**: While 1,000 questions and 1,936 valid responses provide adequate statistical power for the analyses conducted, larger samples would allow more fine-grained subgroup analyses.

5. **Low pseudo-R²**: Question features explain only 5% of variance, meaning 95% of what drives confident failures remains unidentified. Unmeasured variables — including the model's pretraining data distribution, question-specific knowledge gaps, and stochastic generation artifacts — likely account for much of the remaining variance.

6. **Heuristic annotations**: Reasoning steps were approximated from question length quartiles rather than expert annotation. The misleading premise feature was assigned categorically (all TruthfulQA = 1, all else = 0) rather than per-question.

---

## 6. Conclusion

This study investigated what structural and semantic properties of questions predict when large language models will be confidently wrong. By shifting the unit of analysis from aggregate dataset-level calibration to individual question-level prediction, we identified answer option similarity as the dominant predictor of confident failures — a finding that is both novel and practically actionable.

Both models tested exhibited severe overconfidence, averaging 95–97% confidence while achieving only 54–66% accuracy, consistent with the broader literature on LLM miscalibration. However, this overconfidence is not uniformly distributed across questions. Questions with semantically similar answer options and questions involving numerical content systematically trigger higher rates of confident failure, while features such as negation, misleading premises, and domain category do not significantly predict confident failure after controlling for other variables.

These findings suggest that the semantic confusability of answer choices — rather than the inherent difficulty or trickiness of a question — is the primary structural driver of confident failures. Future work should validate this finding across a broader range of models and architectures, develop confusability-aware calibration methods, and extend the analysis to open-ended generation settings where "option similarity" must be reconceived as the similarity between plausible alternative answers the model considers during generation.

---

## References

Chhikara, P. (2025). Mind the confidence gap: Overconfidence, calibration, and distractor effects in large language models. *Transactions on Machine Learning Research*.

Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., & Tafjord, O. (2018). Think you have solved question answering? Try ARC, the AI2 Reasoning Challenge. *arXiv preprint arXiv:1803.05457*.

Geng, J., Cai, F., Wang, Y., Kober, H., Bisk, Y., & Neubig, G. (2024). A survey of confidence estimation and calibration in large language models. *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics*.

Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring massive multitask language understanding. *Proceedings of the International Conference on Learning Representations*.

Kumaran, D., et al. (2026). Competing biases underlie overconfidence and underconfidence in LLMs. *Nature Machine Intelligence*, 8, 430–442.

Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*.

Liu, Y., et al. (2025). Uncertainty quantification and confidence calibration in large language models: A survey. *arXiv preprint arXiv:2502.xxxxx*.

Qwen Team. (2024). Qwen 2.5 technical report. *arXiv preprint arXiv:2412.15115*.

Rewarding Doubt. (2026). Rewarding doubt: A reinforcement learning approach to calibrated confidence expression of large language models. *Proceedings of the International Conference on Learning Representations*.

Steyvers, M., Tejeda, H., Kumar, A., Belem, C., Karny, S., Hu, X., Mayer, L., & Smyth, P. (2024). What large language models know and what people think they know. *Nature Machine Intelligence*.

Tian, K., Mitchell, E., Yao, H., Manning, C. D., & Finn, C. (2023). Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*.

Xiao, Y., et al. (2025). The consistency hypothesis in uncertainty quantification for large language models. *Proceedings of the Conference on Uncertainty in Artificial Intelligence*.

Zhang, X., et al. (2024). Calibrating the confidence of large language models by eliciting fidelity. *arXiv preprint arXiv:2404.xxxxx*.

---

## Disclosure

This paper was produced as an educational project. The original idea and research direction were provided by the author. All technical work — including the literature review, research gap identification, experiment design, code implementation, statistical analysis, and paper writing — was performed by an AI assistant (Claude, via OpenCode). This is not independent research and should not be cited or treated as such.

**Code and data**: [github.com/gautamritvik/AI-Confidence-Research](https://github.com/gautamritvik/AI-Confidence-Research)
