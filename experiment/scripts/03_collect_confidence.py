"""
Step 3: Query each model on every question, collecting answers and 
verbalized confidence scores.

For each (question, model) pair, we get:
  - model_answer (A/B/C/D)
  - model_confidence (0-100)
  - is_correct (bool)
  - raw_response (full text)

Saves results incrementally so interrupted runs can resume.

Input:  data/questions_annotated.parquet
Output: data/responses_{model_name}.parquet  (one per model)
        data/all_responses.parquet           (combined)
"""
import re
import sys
import time
import json
import argparse
import pandas as pd
from pathlib import Path
from config import (DATA_DIR, MODELS, PROMPT_TEMPLATE, OLLAMA_BASE_URL,
                    TEMPERATURE, MAX_RETRIES, RETRY_DELAY, MAX_TOKENS, BATCH_SIZE)


def build_prompt(row: pd.Series) -> str:
    """Build the confidence elicitation prompt for a question."""
    options = (f"A) {row['option_a']}\n"
               f"B) {row['option_b']}\n"
               f"C) {row['option_c']}\n"
               f"D) {row['option_d']}")
    return PROMPT_TEMPLATE.format(question=row["question"], options=options)


def parse_response(text: str) -> tuple[str, int]:
    """
    Parse model response to extract answer letter and confidence score.
    Handles chatty models that embed the answer in longer text.
    Returns (letter, confidence) or ("PARSE_ERROR", -1).
    """
    if not text:
        return "PARSE_ERROR", -1
    text = text.strip()

    # Extract answer letter — try multiple patterns
    answer_match = re.search(r'Answer:\s*\[?([A-Da-d])\]?', text)
    if not answer_match:
        # "the correct answer is B"
        answer_match = re.search(r'(?:correct answer|answer) (?:is|:)\s*\(?([A-Da-d])\)?', text, re.IGNORECASE)
    if not answer_match:
        # "My answer is: B" or similar
        answer_match = re.search(r'(?:My answer|I (?:choose|pick|select))\s*(?:is)?:?\s*\(?([A-Da-d])\)?', text, re.IGNORECASE)
    if not answer_match:
        # Standalone letter at start of line
        answer_match = re.search(r'^([A-Da-d])\b', text, re.MULTILINE)
    if not answer_match:
        # Last resort: any "option X" reference
        answer_match = re.search(r'(?:option|choice)\s+([A-Da-d])\b', text, re.IGNORECASE)
    
    # Extract confidence — try multiple patterns
    conf_match = re.search(r'Confidence:\s*\[?(\d{1,3})\]?', text)
    if not conf_match:
        conf_match = re.search(r'confidence\s*(?:is|:|-|=)\s*\[?(\d{1,3})\]?', text, re.IGNORECASE)
    if not conf_match:
        # "95% confident" or "confidence level: 85"
        conf_match = re.search(r'(\d{1,3})\s*(?:%|percent)\s*confident', text, re.IGNORECASE)
    if not conf_match:
        # Last number in text between 0-100 (heuristic)
        all_nums = re.findall(r'\b(\d{1,3})\b', text)
        valid_nums = [int(n) for n in all_nums if 0 <= int(n) <= 100]
        if valid_nums:
            conf_match = type('Match', (), {'group': lambda self, x: str(valid_nums[-1])})()

    answer = answer_match.group(1).upper() if answer_match else "PARSE_ERROR"
    confidence = int(conf_match.group(1)) if conf_match else -1

    # Clamp confidence to 0-100
    if confidence > 100:
        confidence = 100
    if confidence < 0 and confidence != -1:
        confidence = 0

    return answer, confidence


def query_openai(client, model_id: str, prompt: str) -> str:
    """Query an OpenAI model."""
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content


def collect_for_model(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Collect responses for all questions from one model."""
    model_cfg = MODELS[model_name]
    model_id = model_cfg["model_id"]

    # Initialize Ollama client (local, no API key needed)
    from openai import OpenAI
    client = OpenAI(
        api_key="ollama",  # required but unused by Ollama
        base_url=OLLAMA_BASE_URL,
    )
    query_fn = lambda prompt: query_openai(client, model_id, prompt)

    # Check for existing progress
    progress_path = DATA_DIR / f"_progress_responses_{model_name}.json"
    if progress_path.exists():
        with open(progress_path) as f:
            existing = json.load(f)
        print(f"  Resuming: {len(existing)} already collected")
    else:
        existing = {}

    results = []
    new_queries = 0
    errors = 0
    total = len(df)

    for idx, row in df.iterrows():
        qid = row["question_id"]

        # Use cached result
        if qid in existing:
            results.append(existing[qid])
            continue

        prompt = build_prompt(row)

        for attempt in range(MAX_RETRIES):
            try:
                raw = query_fn(prompt)
                answer, confidence = parse_response(raw)

                result = {
                    "question_id": qid,
                    "model": model_name,
                    "model_answer": answer,
                    "model_confidence": confidence,
                    "is_correct": int(answer == row["correct_letter"]),
                    "raw_response": raw,
                }
                results.append(result)
                existing[qid] = result
                new_queries += 1

                # No rate limit needed for local Ollama
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    # Rate limited — wait longer
                    wait = 10 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                elif attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (2 ** attempt)
                    time.sleep(wait)
                else:
                    print(f"  FAILED {qid}: {e}")
                    result = {
                        "question_id": qid,
                        "model": model_name,
                        "model_answer": "ERROR",
                        "model_confidence": -1,
                        "is_correct": 0,
                        "raw_response": str(e),
                    }
                    results.append(result)
                    existing[qid] = result
                    errors += 1

        # Progress update and save
        done = len(results)
        if new_queries > 0 and (new_queries % BATCH_SIZE == 0 or done == total):
            with open(progress_path, "w") as f:
                json.dump(existing, f)
            elapsed_pct = 100 * done / total
            print(f"  [{model_name}] {done}/{total} ({elapsed_pct:.1f}%) "
                  f"- {new_queries} new, {errors} errors")

    # Final save
    with open(progress_path, "w") as f:
        json.dump(existing, f)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Collect confidence responses from LLMs"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run only one model (e.g., 'gpt-4o-mini'). "
             "Default: run all models."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only first 10 questions to test the pipeline."
    )
    args = parser.parse_args()

    input_path = DATA_DIR / "questions_annotated.parquet"
    if not input_path.exists():
        print("ERROR: Run 02_annotate_features.py first.")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    
    if args.dry_run:
        df = df.head(10)
        print(f"DRY RUN: Using {len(df)} questions")
    else:
        print(f"Loaded {len(df)} questions")

    # Determine which models to run
    models_to_run = [args.model] if args.model else list(MODELS.keys())

    all_responses = []

    for model_name in models_to_run:
        if model_name not in MODELS:
            print(f"ERROR: Unknown model '{model_name}'. "
                  f"Available: {list(MODELS.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"Collecting from: {model_name}")
        print(f"{'='*60}")

        model_df = collect_for_model(df, model_name)

        # Save per-model results
        out_path = DATA_DIR / f"responses_{model_name}.parquet"
        model_df.to_parquet(out_path, index=False)
        print(f"  Saved {len(model_df)} responses to {out_path}")

        # Quick stats
        valid = model_df[model_df["model_confidence"] >= 0]
        if len(valid) > 0:
            acc = valid["is_correct"].mean()
            avg_conf = valid["model_confidence"].mean()
            parse_errors = (model_df["model_answer"] == "PARSE_ERROR").sum()
            confident_wrong = (
                (valid["model_confidence"] >= 80) & (valid["is_correct"] == 0)
            ).sum()
            print(f"  Accuracy: {acc:.3f}")
            print(f"  Avg confidence: {avg_conf:.1f}")
            print(f"  Parse errors: {parse_errors}")
            print(f"  Confident failures (conf>=80 & wrong): {confident_wrong} "
                  f"({100*confident_wrong/len(valid):.1f}%)")

        all_responses.append(model_df)

    # Combine all model responses
    if all_responses:
        combined = pd.concat(all_responses, ignore_index=True)
        out_path = DATA_DIR / "all_responses.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"\nCombined: {len(combined)} total responses saved to {out_path}")


if __name__ == "__main__":
    main()
