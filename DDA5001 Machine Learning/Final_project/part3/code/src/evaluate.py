import json
import argparse
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter
from verifier import compute_score
import math
import random

# Fix random seed for tie-breaking in Majority Vote
random.seed(42)

def evaluate_results(input_file, output_file):
    """
    Reads a JSONL file of model generations, computes scores, and saves the
    enhanced results to a new JSONL file.

    Computes:
    1. Unbiased pass@k estimates.
    2. Majority Vote (@1) accuracy.
    """
    # Group scores by problem id for pass@k
    # Key: problem_id -> Value: list of scores
    problem2scores = defaultdict(list)

    # Group predictions by problem id for Majority Vote
    # Key: problem_id -> Value: list of tuples (extracted_pred, score)
    problem2maj_data = defaultdict(list)

    input_path = Path(input_file)
    if not input_path.is_file():
        print(f"Error: Input file not found at: {input_file}")
        return

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines_processed = 0

    # Pre-calculate total lines for the progress bar
    print(f"Counting lines in {input_file}...")
    with open(input_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    print(f"Scoring generations from {input_file}...")

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line_idx, line in enumerate(tqdm(f_in, total=total_lines, desc="Processing lines")):
            data = json.loads(line)
            lines_processed += 1

            # Basic fields
            model_answer = data.get("answer", "")
            gold_answer = data.get("gold", "")
            
            # Use 'id' to group rollouts for the same problem
            problem_id = data.get("id", line_idx)

            try:
                # Compute correctness
                score_dict = compute_score(model_answer, gold_answer)
                
                # Merge verification results into the data object
                data.update(score_dict)

                if score_dict:
                    score_val = float(score_dict.get("score", 0.0))
                    pred_val = score_dict.get("extracted_pred", None)

                    # 1. Store for Pass@k
                    problem2scores[problem_id].append(score_val)

                    # 2. Store for Majority Vote
                    # We store the prediction to count votes, and the score to verify correctness later
                    problem2maj_data[problem_id].append((pred_val, score_val))

            except Exception as e:
                # Using tqdm.write prevents progress bar breakage
                tqdm.write(f"Error processing line {line_idx + 1}: {e}")
                data["error"] = str(e)

            # Write the line immediately with the new score info
            f_out.write(json.dumps(data) + "\n")

    if lines_processed == 0:
        print("No lines were processed.")
        return

    num_problems = len(problem2scores)
    print(f"Processing complete. Scored {lines_processed} lines across {num_problems} unique problems.")

    # ==========================================
    # 1. Unbiased Pass@k Calculation
    # ==========================================
    k2problem_pass_vals = defaultdict(list)

    for problem_id, scores in problem2scores.items():
        n_resps = len(scores)
        if n_resps == 0:
            continue

        # TODO 1: Count the number of correct responses (c_correct)
        # 'scores' is a list of floats (0.0 or 1.0).
        c_correct = sum(1 for s in scores if s > 0.0)

        # Generate k values: 1, 2, 4, ... up to n_resps
        ks = []
        k = 1
        while k <= n_resps:
            ks.append(k)
            k *= 2
        
        if n_resps not in ks:
            ks.append(n_resps)
        
        ks = sorted(list(set(ks)))

        for k in ks:
            # TODO 2: Implement the unbiased pass@k formula.
            # Formula: pass@k = 1 - combin(n-c, k) / combin(n, k)
            # Hint: Use math.comb(n, k). Handle the edge case where k > n_resps (set to 1.0)
            # and catch potential value errors if needed.
            
            if k > n_resps:
                pass_at_k = 1.0
            else:
                try:
                    pass_at_k = 1.0 - (math.comb(n_resps - c_correct, k) / math.comb(n_resps, k))
                except (ValueError, ZeroDivisionError):
                    pass_at_k = 0.0

            k2problem_pass_vals[k].append(pass_at_k)

    # ==========================================
    # 2. Majority Vote Calculation
    # ==========================================
    maj_correct_count = 0
    maj_total = 0

    for problem_id, data_list in problem2maj_data.items():
        if not data_list:
            continue
        
        maj_total += 1
        
        # data_list is a list of (extracted_pred, score)
        # Extract just the predictions for counting
        preds = [item[0] for item in data_list]
        
        # TODO 3: Implement Majority Vote logic.
        # Steps:
        # a) Count the frequency of each prediction in `preds`.
        # b) Identify the most frequent prediction (the "winner").
        #    Note: If there is a tie, you must break it deterministically 
        #    (e.g., sort candidates) before picking one, or use random.choice 
        #    if you want random tie breaking (we seeded random at the top).
        # c) Check if the winner is correct (is_winner_correct).
        #    You can look up the score associated with the winner in `data_list`.
        
        # 3a. Find the winner
        # Count frequencies (Counter handles None keys fine)
        counts = Counter(preds)
        
        # Find the max frequency
        max_freq = max(counts.values())
        
        # Get all predictions that share the max frequency (handle ties)
        candidates = [p for p, c in counts.items() if c == max_freq]
        
        # Tie-breaking logic
        if len(candidates) == 1:
            winner = candidates[0]
        else:
            # Sort to ensure deterministic behavior before random choice
            # (We convert to str because candidates might contain None or numbers)
            candidates.sort(key=lambda x: str(x))
            winner = random.choice(candidates)
        
        # 3b. Check if winner is correct
        # We look up the score associated with this winner from our stored data.
        # (Assuming the same extracted prediction always yields the same correctness)
        is_winner_correct = False
        for pred, score in data_list:
            if pred == winner:
                if score > 0.0:
                    is_winner_correct = True
                break
        
        if is_winner_correct:
            maj_correct_count += 1

    # ==========================================
    # Print Results
    # ==========================================
    
    # Print Pass@k
    if k2problem_pass_vals:
        print("\nPass@k Metrics:")
        for k in sorted(k2problem_pass_vals.keys()):
            vals = k2problem_pass_vals[k]
            avg_pass = (sum(vals) / len(vals)) * 100.0
            print(f"  pass@{k:<4}: {avg_pass:.2f}%")
    else:
        print("No results found for Pass@k computation.")

    # Print Majority Vote
    if maj_total > 0:
        maj_acc = (maj_correct_count / maj_total) * 100.0
        print("\nMajority Vote Metric:")
        print(f"  maj@1    : {maj_acc:.2f}%")

    print(f"\nScored results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model generations (Pass@k and Majority Vote)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="The path to the input JSONL file (generations).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The path to the output JSONL file (scored results).",
    )
    args = parser.parse_args()

    evaluate_results(args.input_file, args.output_file)