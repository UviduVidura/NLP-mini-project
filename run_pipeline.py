"""
Master NLP Pipeline Runner
==========================
Orchestrates all steps of the NLP Mini Project:
  Step 0: Scrape ~5,000 GSMArena reviews (scrape_gsmarena_5k.py)
  Step 1: Preprocessing, VADER 3-class labeling, and Baseline ML (1_preprocess_and_label.py)
  Step 2: BERT Transformer Fine-Tuning (2_train_bert.py)
  Step 3: Comparative Evaluation & Error Analysis (3_compare_and_evaluate.py)

Usage:
  python run_pipeline.py --all           # Runs entire pipeline from scratch
  python run_pipeline.py --step 1        # Runs preprocessing & baselines
  python run_pipeline.py --step 2        # Runs BERT fine-tuning
  python run_pipeline.py --step 3        # Runs comparison & error analysis report
"""

import sys
import subprocess
import argparse

STEPS = {
    0: ("Web Scraper (~5k GSMArena Reviews)", "scrape_gsmarena_5k.py"),
    1: ("Data Preprocessing, VADER Labeling & Baseline ML", "1_preprocess_and_label.py"),
    2: ("BERT Transformer Fine-Tuning", "2_train_bert.py"),
    3: ("Comparative Evaluation & Error Analysis", "3_compare_and_evaluate.py"),
}


def run_script(step_num):
    desc, script = STEPS[step_num]
    print("\n" + "=" * 75)
    print(f"EXECUTING STEP {step_num}: {desc} ({script})")
    print("=" * 75)

    cmd = [sys.executable, script]
    try:
        ret = subprocess.run(cmd, check=True)
        print(f"Step {step_num} finished successfully with exit code {ret.returncode}.")
    except subprocess.CalledProcessError as e:
        print(f"\nError executing Step {step_num}: {e}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="NLP Mini Project Master Runner")
    parser.add_argument("--step", type=int, choices=[0, 1, 2, 3], help="Specific step to run")
    parser.add_argument("--all", action="store_true", help="Run the entire end-to-end pipeline (steps 0 to 3)")
    parser.add_argument("--skip-scraping", action="store_true", help="Run steps 1, 2, and 3, using existing reviews data")

    args = parser.parse_args()

    if args.step is not None:
        run_script(args.step)
    elif args.skip_scraping:
        for s in [1, 2, 3]:
            run_script(s)
    elif args.all:
        for s in [0, 1, 2, 3]:
            run_script(s)
    else:
        print("Please specify a command. Examples:")
        print("  python run_pipeline.py --skip-scraping   # Uses existing review data, runs steps 1-3")
        print("  python run_pipeline.py --all             # Runs live scraper then all models")
        print("  python run_pipeline.py --step 1          # Runs preprocessing & baselines only")
        print("  python run_pipeline.py --step 3          # Generates comparison report")


if __name__ == "__main__":
    main()
