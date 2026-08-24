"""
Central configuration for the experiment.
Question-Level Predictors of Confident Failures in LLMs.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── API Keys ───────────────────────────────────────────────────────────
# Set this as an environment variable before running:
#   export OPENROUTER_API_KEY="sk-or-..."
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────
# All models routed through OpenRouter (https://openrouter.ai)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS = {
    "llama-3.1-8b": {
        "provider": "ollama",
        "model_id": "llama3.1:8b",
    },
    "qwen-2.5-3b": {
        "provider": "ollama",
        "model_id": "qwen2.5:3b",
    },
    "phi-3-3.8b": {
        "provider": "ollama",
        "model_id": "phi3:3.8b",
    },
}

# ── Ollama ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# ── Sample Size ────────────────────────────────────────────────────────
SAMPLE_SIZE = 1000  # stratified sample from full 16k dataset

# ── Experiment Parameters ──────────────────────────────────────────────
CONFIDENCE_SCALE = (0, 100)
CONFIDENT_THRESHOLD = 80          # confidence >= this AND wrong = "confident failure"
TEMPERATURE = 0.0                 # deterministic for main run
CONSISTENCY_TEMPERATURE = 0.7     # for consistency sub-sample
CONSISTENCY_SAMPLES = 5           # number of repeated queries for consistency check
CONSISTENCY_FRACTION = 0.10       # fraction of questions for consistency check
MAX_RETRIES = 3                   # retries on API failure
RETRY_DELAY = 2                   # seconds between retries
BATCH_SIZE = 50                   # questions per batch for progress tracking
MAX_TOKENS = 50                   # max tokens for model response (answer + confidence)

# ── Datasets ───────────────────────────────────────────────────────────
DATASETS = {
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_config": "all",
        "split": "test",
    },
    "arc_challenge": {
        "hf_path": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "split": "test",
    },
    "truthfulqa": {
        "hf_path": "truthful_qa",
        "hf_config": "multiple_choice",
        "split": "validation",
    },
}

# ── Prompt Template ────────────────────────────────────────────────────
PROMPT_TEMPLATE = """Answer the following multiple-choice question. After providing your answer, rate your confidence that your answer is correct on a scale from 0 to 100, where 0 means you are certain you are wrong and 100 means you are certain you are correct.

Question: {question}
{options}

Respond in exactly this format:
Answer: [A/B/C/D]
Confidence: [0-100]"""

# ── Feature Annotation ─────────────────────────────────────────────────
REASONING_STEPS_PROMPT = """Analyze the following multiple-choice question and determine the minimum number of distinct logical/reasoning steps needed to arrive at the correct answer.

Count only essential reasoning steps, not reading or comprehension steps. Examples:
- Direct fact recall = 1 step
- Fact recall + application = 2 steps  
- Multi-step calculation or inference chain = 3+ steps

Question: {question}
Options: {options}
Correct Answer: {correct_answer}

Respond with ONLY a single integer (1, 2, 3, or 4 for 4 or more steps):"""

MISLEADING_PREMISE_PROMPT = """Does the following question contain a misleading premise, popular misconception, false presupposition, trick wording, or adversarial framing that could lead someone to choose a wrong answer?

Question: {question}
Options: {options}

Respond with ONLY: YES or NO"""
