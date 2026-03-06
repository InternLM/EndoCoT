import json
import argparse
import re
from pathlib import Path

def parse_sudoku_solution(raw_str):
    if not raw_str or not isinstance(raw_str, str):
        return ""
    
    target_marker = "Solution:"
    idx = raw_str.rfind(target_marker)
    if idx != -1:
        raw_str = raw_str[idx + len(target_marker):]

    digits = re.sub(r'\D', '', raw_str)
    return digits

def count_mismatches(str1, str2):
    if len(str1) != len(str2):
        return abs(len(str1) - len(str2))
    return sum(1 for a, b in zip(str1, str2) if a != b)

def normalize_key(key):
    if key.endswith(".png"):
        return key[:-4]
    return key

def evaluate(gt_path, pred_path, verbose=False):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        raw_pred_data = json.load(f)
        
    pred_map = {normalize_key(k): v for k, v in raw_pred_data.items()}

    total_samples = 0
    correct_count = 0
    missing_count = 0
    invalid_format_count = 0
    wrong_value_count = 0
    
    failed_examples = []
    
    keys = sorted(gt_data.keys())
    
    if verbose:
        print(f"{'Sample ID':<20} | {'Status':<15} | {'Details':<20}")
        print("-" * 65)
    
    for key in keys:
        total_samples += 1
        gt_clean = parse_sudoku_solution(gt_data[key])
        
        key_norm = normalize_key(key)
        pred_raw = pred_map.get(key_norm)
        
        if pred_raw is None:
            missing_count += 1
            failed_examples.append({"id": key, "reason": "MISSING", "gt": gt_clean, "pred": "N/A"})
            if verbose: print(f"{key:<20} | MISSING         | No prediction found")
            continue
            
        pred_clean = parse_sudoku_solution(pred_raw)
        
        if len(pred_clean) != 81:
            invalid_format_count += 1
            failed_examples.append({"id": key, "reason": f"INVALID_LEN ({len(pred_clean)})", "gt": gt_clean, "pred": pred_clean})
            if verbose: print(f"{key:<20} | INVALID_LEN     | Length: {len(pred_clean)}")
        
        elif pred_clean == gt_clean:
            correct_count += 1
            if verbose and total_samples <= 5:
                print(f"{key:<20} | PASS            | Exact Match")
            
        else:
            wrong_value_count += 1
            mismatches = count_mismatches(pred_clean, gt_clean)
            failed_examples.append({"id": key, "reason": f"WRONG ({mismatches} diffs)", "gt": gt_clean, "pred": pred_clean})
            if verbose: print(f"{key:<20} | WRONG           | Mismatched cells: {mismatches}")

    accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0
    print("\n" + "="*50)
    print(f"{'Sudoku Evaluation Results':^50}")
    print("="*50)
    print(f"Total Samples:      {total_samples}")
    print(f"Correct Predictions:{correct_count}")
    print(f"Accuracy:           {accuracy:.2f}%")
    print("-" * 50)
    print(f"Missing Keys:       {missing_count}")
    print(f"Invalid Length:     {invalid_format_count} (Not 81 digits)")
    print(f"Wrong Values:       {wrong_value_count} (81 digits but incorrect)")
    print("="*50)

    if failed_examples:
        num_to_show = 3
        print(f"\n>>> DEBUG: FAILED EXAMPLES (Top {min(num_to_show, len(failed_examples))})")
        for i, ex in enumerate(failed_examples[:num_to_show]):
            print(f"\n[{i+1}] ID: {ex['id']}")
            print(f"    Reason: {ex['reason']}")
            print(f"    GT:   {ex['gt'][:27]}...") 
            print(f"    Pred: {ex['pred'][:27]}...")
            if ex['pred'] != "N/A" and len(ex['pred']) == 81:
                for idx, (a, b) in enumerate(zip(ex['gt'], ex['pred'])):
                    if a != b:
                        print(f"    First error at index {idx}: Expected '{a}', got '{b}'")
                        break
        print("\n" + "="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sudoku Predictions")
    parser.add_argument("gt", type=str, help="Path to Ground Truth JSON")
    parser.add_argument("pred", type=str, help="Path to Prediction JSON")
    parser.add_argument("--verbose", action="store_true", help="Print status for every sample")
    
    args = parser.parse_args()
    
    if not Path(args.gt).exists() or not Path(args.pred).exists():
        print("Error: Input files not found.")
    else:
        evaluate(args.gt, args.pred, args.verbose)