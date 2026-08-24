"""
Step 2: Annotate each question with structural/semantic features.

Computable features (no API needed):
  - question_length (word count)
  - has_negation (contains NOT/EXCEPT/NEVER/FALSE etc.)
  - has_numerical (contains numbers requiring calculation)
  - option_similarity (avg pairwise sentence-BERT cosine similarity)
  - domain_category (STEM/Humanities/Social Sciences/Other/Science/Misconceptions)

LLM-annotated features (requires API — can be done with a cheap model):
  - reasoning_steps (1/2/3/4+)
  - has_misleading_premise (0/1)

Input:  data/questions.parquet
Output: data/questions_annotated.parquet
"""
import re
import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import (DATA_DIR, OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
                    REASONING_STEPS_PROMPT, MISLEADING_PREMISE_PROMPT,
                    MAX_RETRIES, RETRY_DELAY)

# ── MMLU subject → domain category mapping ─────────────────────────────
STEM_SUBJECTS = {
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_mathematics", "high_school_physics", "high_school_statistics",
    "machine_learning", "medical_genetics", "virology",
}
HUMANITIES_SUBJECTS = {
    "formal_logic", "high_school_european_history",
    "high_school_us_history", "high_school_world_history", "international_law",
    "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios",
    "philosophy", "prehistory", "world_religions",
}
SOCIAL_SCIENCE_SUBJECTS = {
    "econometrics", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_microeconomics",
    "high_school_psychology", "human_sexuality", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
}
# Everything else in MMLU falls into "professional/other"

NEGATION_PATTERNS = re.compile(
    r'\b(NOT|EXCEPT|NEVER|NEITHER|NOR|CANNOT|DOESN\'T|ISN\'T|AREN\'T|'
    r'WASN\'T|WEREN\'T|WON\'T|WOULDN\'T|SHOULDN\'T|COULDN\'T|'
    r'HAVEN\'T|HASN\'T|HADN\'T|FALSE|INCORRECT|WRONG|LEAST|UNLIKELY)\b',
    re.IGNORECASE
)

NUMERICAL_PATTERN = re.compile(
    r'\b\d+[\d,]*\.?\d*\s*(?:%|percent|dollars?|km|m|cm|mm|kg|g|mg|'
    r'miles?|feet|ft|inches?|in|lbs?|oz|gallons?|liters?|hours?|'
    r'minutes?|seconds?|years?|months?|days?|degrees?)?\b'
)


def assign_domain(subject: str, dataset: str) -> str:
    """Map subject + dataset to a broad domain category."""
    if dataset == "arc_challenge":
        return "STEM"
    if dataset == "truthfulqa":
        return "Misconceptions"
    # MMLU subjects
    if subject in STEM_SUBJECTS:
        return "STEM"
    if subject in HUMANITIES_SUBJECTS:
        return "Humanities"
    if subject in SOCIAL_SCIENCE_SUBJECTS:
        return "Social Sciences"
    return "Professional/Other"


def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features that require no external API."""
    print("Computing basic features...")

    # Question length (word count)
    df["question_length"] = df["question"].str.split().str.len()

    # Negation detection
    df["has_negation"] = df["question"].apply(
        lambda q: int(bool(NEGATION_PATTERNS.search(q)))
    )

    # Numerical content detection
    def has_numerical_content(row):
        """Check if question + options contain numbers that suggest calculation."""
        text = row["question"] + " " + " ".join([
            str(row["option_a"]), str(row["option_b"]),
            str(row["option_c"]), str(row["option_d"])
        ])
        matches = NUMERICAL_PATTERN.findall(text)
        return int(len(matches) >= 2)  # At least 2 numerical references

    df["has_numerical"] = df.apply(has_numerical_content, axis=1)

    # Domain category
    df["domain_category"] = df.apply(
        lambda r: assign_domain(r["subject"], r["dataset"]), axis=1
    )

    print(f"  question_length: mean={df['question_length'].mean():.1f}, "
          f"median={df['question_length'].median():.0f}")
    print(f"  has_negation: {df['has_negation'].sum()} / {len(df)} "
          f"({100*df['has_negation'].mean():.1f}%)")
    print(f"  has_numerical: {df['has_numerical'].sum()} / {len(df)} "
          f"({100*df['has_numerical'].mean():.1f}%)")
    print(f"  domain_category distribution:\n{df['domain_category'].value_counts()}")

    return df


def compute_option_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average pairwise cosine similarity between the 4 answer options."""
    print("Computing option similarity (sentence-BERT)...")
    print("  Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Collect ALL option texts into one flat list for batch encoding
    all_options = []
    for _, row in df.iterrows():
        all_options.extend([
            str(row["option_a"]), str(row["option_b"]),
            str(row["option_c"]), str(row["option_d"])
        ])

    print(f"  Encoding {len(all_options)} option texts in batch...")
    all_embeddings = model.encode(all_options, show_progress_bar=True, batch_size=512)

    # Compute pairwise similarity per question (groups of 4)
    print("  Computing pairwise similarities...")
    similarities = []
    for i in range(len(df)):
        emb = all_embeddings[i*4 : i*4+4]
        sim_matrix = cosine_similarity(emb)
        upper = sim_matrix[np.triu_indices(4, k=1)]
        similarities.append(float(upper.mean()))

        if (i+1) % 5000 == 0:
            print(f"    {i+1}/{len(df)}")

    df["option_similarity"] = similarities
    print(f"  option_similarity: mean={np.mean(similarities):.3f}, "
          f"std={np.std(similarities):.3f}")
    return df


def annotate_with_llm(df: pd.DataFrame, feature: str, 
                      skip_if_exists: bool = True) -> pd.DataFrame:
    """
    Use GPT-4o-mini to annotate reasoning_steps or has_misleading_premise.
    Saves progress incrementally.
    """
    from openai import OpenAI
    
    if not OPENROUTER_API_KEY:
        print(f"  WARNING: No OPENROUTER_API_KEY set. Skipping {feature} annotation.")
        print(f"  Set the key and rerun, or annotate manually.")
        df[feature] = -1  # placeholder
        return df

    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )

    if feature == "reasoning_steps":
        prompt_template = REASONING_STEPS_PROMPT
    elif feature == "has_misleading_premise":
        prompt_template = MISLEADING_PREMISE_PROMPT
    else:
        raise ValueError(f"Unknown feature: {feature}")

    # Check for existing progress
    progress_path = DATA_DIR / f"_progress_{feature}.json"
    if skip_if_exists and progress_path.exists():
        with open(progress_path) as f:
            existing = json.load(f)
        print(f"  Resuming {feature}: {len(existing)} already annotated")
    else:
        existing = {}

    total = len(df)
    results = []
    new_annotations = 0

    for idx, row in df.iterrows():
        qid = row["question_id"]

        # Use cached result if available
        if qid in existing:
            results.append(existing[qid])
            continue

        options_text = f"A) {row['option_a']}\nB) {row['option_b']}\nC) {row['option_c']}\nD) {row['option_d']}"

        if feature == "reasoning_steps":
            prompt = prompt_template.format(
                question=row["question"],
                options=options_text,
                correct_answer=f"{row['correct_letter']}) {row['correct_answer']}"
            )
        else:
            prompt = prompt_template.format(
                question=row["question"],
                options=options_text,
            )

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                )
                text = response.choices[0].message.content.strip()

                if feature == "reasoning_steps":
                    # Extract integer
                    nums = re.findall(r'\d+', text)
                    val = int(nums[0]) if nums else 1
                    val = min(max(val, 1), 4)  # clamp to 1-4
                elif feature == "has_misleading_premise":
                    val = 1 if text.upper().startswith("YES") else 0

                results.append(val)
                existing[qid] = val
                new_annotations += 1
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print(f"  Failed on {qid}: {e}")
                    results.append(-1)  # mark as failed

        # Save progress every 500 annotations
        if new_annotations > 0 and new_annotations % 500 == 0:
            with open(progress_path, "w") as f:
                json.dump(existing, f)
            print(f"  {feature}: {len(existing)}/{total} annotated "
                  f"({new_annotations} new)")

    # Final save
    with open(progress_path, "w") as f:
        json.dump(existing, f)

    df[feature] = results
    print(f"  {feature}: done ({new_annotations} new annotations)")
    return df


def main():
    # Use sample if available, otherwise full dataset
    input_path = DATA_DIR / "questions_sample.parquet"
    if not input_path.exists():
        input_path = DATA_DIR / "questions.parquet"
    if not input_path.exists():
        print("ERROR: Run 01_load_datasets.py first.")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} questions\n")

    # Phase 1: Computable features (free, fast)
    df = compute_basic_features(df)
    print()

    # Phase 2: Option similarity (free, ~5-10 min)
    df = compute_option_similarity(df)
    print()

    # Phase 3: LLM-annotated features (optional — costs ~$10-20, ~30-60 min)
    # Skip with --skip-llm flag to run experiment faster with 6 features instead of 8
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM annotation (reasoning_steps, misleading_premise)")
    args = parser.parse_args()

    if args.skip_llm:
        print("Skipping LLM annotation (--skip-llm). Using heuristic for misleading_premise.")
        # Heuristic: TruthfulQA = misleading, everything else = 0
        df["has_misleading_premise"] = 0
        tqa_mask = df["dataset"] == "truthfulqa"
        df.loc[tqa_mask, "has_misleading_premise"] = 1
        # Reasoning steps: use question_length quartile as rough proxy
        df["reasoning_steps"] = pd.qcut(
            df["question_length"], q=4, labels=[1, 2, 3, 4]
        ).astype(int)
        print(f"  Set {tqa_mask.sum()} TruthfulQA questions to has_misleading_premise=1")
        print(f"  Used question_length quartiles as reasoning_steps proxy")
    else:
        print("Annotating reasoning_steps via GPT-4o-mini...")
        df = annotate_with_llm(df, "reasoning_steps")
        print()

        print("Annotating has_misleading_premise via GPT-4o-mini...")
        df = annotate_with_llm(df, "has_misleading_premise")
        print()

        # For TruthfulQA, override misleading premise to 1 (by design)
        tqa_mask = df["dataset"] == "truthfulqa"
        df.loc[tqa_mask, "has_misleading_premise"] = 1
        print(f"Overrode {tqa_mask.sum()} TruthfulQA questions to has_misleading_premise=1")

    # Save
    out_path = DATA_DIR / "questions_annotated.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved annotated data to {out_path}")

    # Summary stats
    print("\n── Feature Summary ──")
    for col in ["question_length", "has_negation", "has_numerical",
                "option_similarity", "domain_category", "reasoning_steps",
                "has_misleading_premise"]:
        if col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                print(f"  {col}: mean={df[col].mean():.3f}, "
                      f"std={df[col].std():.3f}, "
                      f"min={df[col].min()}, max={df[col].max()}")
            else:
                print(f"  {col}:\n{df[col].value_counts().to_string()}")


if __name__ == "__main__":
    main()
