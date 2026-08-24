"""
Step 1b: Draw a stratified random sample from the full dataset.

Stratified by dataset (MMLU/ARC/TruthfulQA) proportionally,
ensuring all MMLU domains are represented.

Input:  data/questions.parquet (16,006 questions)
Output: data/questions_sample.parquet (1,000 questions)
"""
import pandas as pd
from config import DATA_DIR, SAMPLE_SIZE

def main():
    df = pd.read_parquet(DATA_DIR / "questions.parquet")
    print(f"Full dataset: {len(df)} questions")

    # Proportional stratified sample by dataset
    # MMLU: 14042/16006 = 87.7% → ~877
    # ARC:  1147/16006  = 7.2%  → ~72
    # TQA:  817/16006   = 5.1%  → ~51
    # But we want enough TQA for the misleading premise analysis,
    # so we'll oversample TQA slightly and keep ARC full-ish.

    n_mmlu = 800
    n_arc = 100
    n_tqa = 100  # oversample relative to proportion for statistical power

    # Within MMLU, stratify by domain_category proxy (subject)
    mmlu = df[df["dataset"] == "mmlu"]
    # Sample proportionally across subjects
    mmlu_sample = mmlu.groupby("subject", group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max(1, round(n_mmlu * len(x) / len(mmlu)))),
                           random_state=42)
    )
    # Trim or pad to exact n_mmlu
    if len(mmlu_sample) > n_mmlu:
        mmlu_sample = mmlu_sample.sample(n=n_mmlu, random_state=42)
    
    arc = df[df["dataset"] == "arc_challenge"].sample(
        n=min(n_arc, len(df[df["dataset"] == "arc_challenge"])), random_state=42
    )
    tqa = df[df["dataset"] == "truthfulqa"].sample(
        n=min(n_tqa, len(df[df["dataset"] == "truthfulqa"])), random_state=42
    )

    sample = pd.concat([mmlu_sample, arc, tqa], ignore_index=True)
    sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"\nSample: {len(sample)} questions")
    print(f"  MMLU: {len(mmlu_sample)} across {mmlu_sample['subject'].nunique()} subjects")
    print(f"  ARC:  {len(arc)}")
    print(f"  TQA:  {len(tqa)}")

    out_path = DATA_DIR / "questions_sample.parquet"
    sample.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
