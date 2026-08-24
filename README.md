# AI-Confidence-Research

A research project investigating what structural and semantic properties of questions cause large language models to be confidently wrong — studying confidence miscalibration across multiple LLMs and question types using MMLU, ARC-Challenge, and TruthfulQA.

## Research Question

What structural and semantic properties of questions predict when large language models will be confidently wrong?

## Structure

```
literature/          # Papers, notes, and literature matrix
experiment/          # All experiment code
  scripts/           # Local pipeline (data loading, annotation, analysis)
  colab_experiment.ipynb  # Google Colab notebook (runs on free GPU)
  data/              # Generated datasets and responses
  results/           # Statistical analysis outputs
  figures/           # Generated plots
research/            # Research question notes and planning
```

## Datasets

- MMLU (800 questions, 57 subjects)
- ARC-Challenge (100 questions)
- TruthfulQA (100 questions)

## Models Evaluated

- Qwen 2.5 3B Instruct
- Qwen 2.5 1.5B Instruct

## Key Features Analyzed

- Question length
- Presence of negation
- Numerical content
- Answer option similarity
- Domain category
- Misleading premise
- Estimated reasoning steps

## Methods

- Verbalized confidence elicitation (0–100 scale)
- Logistic regression predicting confident failure
- Random forest feature importance
- Expected Calibration Error (ECE) per domain and model
- Cross-model failure profile comparison

## Literature

9 papers reviewed covering LLM confidence estimation, calibration, overconfidence, uncertainty quantification, and distractor effects. Full literature matrix in `literature/literature_matrix.csv`.

## Status

Data collection complete (2,000 responses across 2 models). Analysis and paper writing in progress.
