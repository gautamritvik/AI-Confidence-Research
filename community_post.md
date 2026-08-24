# I used an AI to research when LLMs are "confidently wrong" — here's what we found

**Full disclosure**: I came up with the original idea and direction for this project, but an AI assistant (Claude) did the actual research work — the literature review, experiment design, code, statistical analysis, and paper writing. I set the constraints (free models only, what datasets to use, etc.) and made decisions along the way, but I want to be upfront that I didn't write the code or conduct the analysis myself. This was a learning project to understand how AI research actually works.

---

An AI assistant ran an experiment to figure out what makes LLMs answer questions with high confidence but get them wrong. Instead of just measuring overall calibration like most papers do, we looked at what properties of individual questions predict confident failures.

## Setup

- **1,000 questions** from MMLU (57 subjects), ARC-Challenge (science reasoning), and TruthfulQA (misconceptions)
- **2 models**: Qwen 2.5 3B and Qwen 2.5 1.5B
- Each model answers the question and gives a confidence score (0-100)
- A "confident failure" = wrong answer + confidence >= 80%

Each question was annotated with features like:
- Answer option similarity (how semantically close the 4 choices are, using sentence-BERT)
- Numerical content
- Negation (NOT, EXCEPT, etc.)
- Misleading premises
- Domain (STEM, humanities, social sciences, etc.)
- Question length

## Key Results

Both models were **massively overconfident**:

| Model | Accuracy | Avg Confidence | ECE |
|-------|----------|---------------|-----|
| Qwen 2.5 3B | 65.5% | 96.9% | 0.349 |
| Qwen 2.5 1.5B | 54.3% | 94.9% | 0.407 |

The 1.5B model is right 54% of the time but acts like it's right 95% of the time.

**39.8% of all responses were confident failures** — wrong answers given with 80%+ confidence.

## The big finding: option similarity

The #1 predictor of confident failure is **how similar the answer choices are to each other**.

When the 4 answer options are semantically close (measured by cosine similarity of sentence-BERT embeddings), the odds of a confident failure increase **nearly 4x** (OR = 3.99, p < 0.0001).

This makes intuitive sense: if the options all sound alike, the model picks one confidently without recognizing it can't actually tell them apart.

Random forest feature importance confirmed this — option similarity accounts for **42% of total importance**, more than every other feature combined.

![Feature Importance](https://raw.githubusercontent.com/gautamritvik/AI-Confidence-Research/main/experiment/figures/fig4_feature_importance.png)

## Other findings

- **Numerical content** increases confident failure odds by 55% (p < 0.001) — models don't know that they can't do math, so they fail confidently
- **Bigger model = fewer failures** — the 3B model had 39% lower odds of confident failure than the 1.5B
- **Negation, misleading premises, and domain were NOT significant** — confident failures are driven by option confusability, not trick questions

## Calibration curves

Both models cluster their confidence near 100% regardless of whether they're right:

![Calibration Curves](https://raw.githubusercontent.com/gautamritvik/AI-Confidence-Research/main/experiment/figures/fig1_calibration_curves.png)

## What this means

LLM overconfidence isn't random. It's predictable. If you can compute the similarity between answer options before asking a model, you can flag questions where its confidence should be discounted.

This could be useful for RAG systems, AI tutoring, or any application where users rely on model confidence.

## Code & Paper

Everything is open source:

- **GitHub**: [github.com/gautamritvik/AI-Confidence-Research](https://github.com/gautamritvik/AI-Confidence-Research)
- Full paper (PDF), experiment code, Colab notebook, literature review, and all results/figures included
- Ran entirely on a free Google Colab T4 GPU

## How this was made

I want to be completely transparent about this. My role was:
- Coming up with the original idea (studying LLM confidence/overconfidence)
- Setting constraints (free models, budget, platform choices)
- Making directional decisions (which research question, which datasets)
- Learning from the process

The AI assistant (Claude, via OpenCode) did:
- Reading and analyzing all 9 research papers
- Identifying the research gap
- Designing the full experiment
- Writing all Python code and the Colab notebook
- Debugging every error
- Running the statistical analysis
- Writing the research paper
- Setting up the GitHub repo

This is NOT my own independent research. It's an AI-assisted learning project. I'm sharing it because the findings are interesting and the process taught me a lot about how research works — but credit for the technical and intellectual work belongs to the AI tool.

Feedback welcome!
