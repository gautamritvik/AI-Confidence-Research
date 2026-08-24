"""
Step 1: Load and normalize MMLU, ARC-Challenge, and TruthfulQA into a 
unified format, then save as a single parquet file.

Output: data/questions.parquet
Columns: question_id, dataset, subject, question, option_a, option_b, 
         option_c, option_d, correct_answer, correct_letter
"""
import pandas as pd
from datasets import load_dataset
from config import DATA_DIR

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_mmlu() -> pd.DataFrame:
    """Load MMLU test split. ~14k questions across 57 subjects."""
    print("Loading MMLU...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    for i, item in enumerate(ds):
        choices = item["choices"]
        correct_idx = item["answer"]
        # Pad to 4 options if needed (MMLU should always have 4)
        while len(choices) < 4:
            choices.append("")
        rows.append({
            "question_id": f"mmlu_{i:05d}",
            "dataset": "mmlu",
            "subject": item.get("subject", "unknown"),
            "question": item["question"],
            "option_a": choices[0],
            "option_b": choices[1],
            "option_c": choices[2],
            "option_d": choices[3],
            "correct_answer": choices[correct_idx],
            "correct_letter": LETTER_MAP[correct_idx],
        })
    df = pd.DataFrame(rows)
    print(f"  MMLU: {len(df)} questions across {df['subject'].nunique()} subjects")
    return df


def load_arc_challenge() -> pd.DataFrame:
    """Load ARC-Challenge test split. ~1.2k reasoning-heavy questions."""
    print("Loading ARC-Challenge...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for i, item in enumerate(ds):
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]

        # Normalize labels to A/B/C/D
        label_to_text = dict(zip(labels, texts))
        # Some ARC questions use 1/2/3/4 instead of A/B/C/D
        letter_map_arc = {"1": "A", "2": "B", "3": "C", "4": "D"}
        
        options = []
        for letter in ["A", "B", "C", "D"]:
            if letter in label_to_text:
                options.append(label_to_text[letter])
            elif letter_map_arc.get(letter, "") in label_to_text:
                options.append(label_to_text[letter_map_arc[letter]])
            else:
                # Some ARC questions have 3 or 5 options; skip non-4-option
                break
        
        if len(options) != 4:
            continue  # Only keep 4-option questions for consistency

        answer_key = item["answerKey"]
        if answer_key in letter_map_arc:
            answer_key = letter_map_arc[answer_key]
        
        correct_idx = ord(answer_key) - ord("A")
        if correct_idx < 0 or correct_idx >= 4:
            continue

        rows.append({
            "question_id": f"arc_{i:05d}",
            "dataset": "arc_challenge",
            "subject": "science",
            "question": item["question"],
            "option_a": options[0],
            "option_b": options[1],
            "option_c": options[2],
            "option_d": options[3],
            "correct_answer": options[correct_idx],
            "correct_letter": answer_key,
        })
    df = pd.DataFrame(rows)
    print(f"  ARC-Challenge: {len(df)} questions (4-option only)")
    return df


def load_truthfulqa() -> pd.DataFrame:
    """Load TruthfulQA multiple-choice split. ~817 misconception questions."""
    print("Loading TruthfulQA...")
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice",
                      split="validation")
    rows = []
    for i, item in enumerate(ds):
        mc1 = item["mc1_targets"]
        choices = mc1["choices"]
        labels = mc1["labels"]

        # mc1: exactly one correct answer (label=1), rest are wrong (label=0)
        correct_idx = labels.index(1) if 1 in labels else None
        if correct_idx is None:
            continue

        # Take first 4 options (correct + up to 3 distractors)
        # Ensure correct answer is included
        if len(choices) < 4:
            # Pad with empty if fewer than 4
            while len(choices) < 4:
                choices.append("[No option]")
                labels.append(0)

        # If correct answer is not in first 4, swap it in
        if correct_idx >= 4:
            choices[3], choices[correct_idx] = choices[correct_idx], choices[3]
            labels[3], labels[correct_idx] = labels[correct_idx], labels[3]
            correct_idx = 3

        # Take first 4
        options = choices[:4]
        correct_letter = LETTER_MAP[correct_idx]

        rows.append({
            "question_id": f"tqa_{i:05d}",
            "dataset": "truthfulqa",
            "subject": "misconceptions",
            "question": item["question"],
            "option_a": options[0],
            "option_b": options[1],
            "option_c": options[2],
            "option_d": options[3],
            "correct_answer": options[correct_idx],
            "correct_letter": correct_letter,
        })
    df = pd.DataFrame(rows)
    print(f"  TruthfulQA: {len(df)} questions")
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    mmlu = load_mmlu()
    arc = load_arc_challenge()
    tqa = load_truthfulqa()

    combined = pd.concat([mmlu, arc, tqa], ignore_index=True)
    print(f"\nTotal: {len(combined)} questions")
    print(f"  MMLU: {len(mmlu)}")
    print(f"  ARC:  {len(arc)}")
    print(f"  TQA:  {len(tqa)}")
    print(f"  Subjects: {combined['subject'].nunique()}")

    out_path = DATA_DIR / "questions.parquet"
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Also save a small CSV preview
    combined.head(20).to_csv(DATA_DIR / "questions_preview.csv", index=False)


if __name__ == "__main__":
    main()
