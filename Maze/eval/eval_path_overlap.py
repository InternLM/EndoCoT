import os
import json
import argparse
from pathlib import Path

def calculate_overlap_rate(gt_json_path, pred_json_path):
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(pred_json_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    results = {}
    total_overlap_ratio = 0
    match_count = 0

    for gt_key, gt_path in gt_data.items():
        clean_key = gt_key.replace(".png", "")
        
        pred_path = None
        if clean_key in pred_data:
            pred_path = pred_data[clean_key]
        elif f"{clean_key}.png" in pred_data:
            pred_path = pred_data[f"{clean_key}.png"]

        if pred_path is not None:
            overlap_len = 0
            min_len = min(len(gt_path), len(pred_path))
            
            for i in range(min_len):
                if gt_path[i] == pred_path[i]:
                    overlap_len += 1
                else:
                    break

            gt_len = len(gt_path)
            ratio = overlap_len / gt_len if gt_len > 0 else 0
            
            results[clean_key] = {
                "overlap_len": overlap_len,
                "gt_len": gt_len,
                "ratio": ratio
            }
            
            total_overlap_ratio += ratio
            match_count += 1

    avg_overlap_rate = total_overlap_ratio / match_count if match_count > 0 else 0

    print("\n" + "="*40)
    print(f"{'ID':<15} | {'Overlap':<8} | {'GT_Len':<8} | {'Ratio':<8}")
    print("-"*40)
    for i, (k, v) in enumerate(results.items()):
        if i < 10:
            print(f"{k:<15} | {v['overlap_len']:<8} | {v['gt_len']:<8} | {v['ratio']:.2%}")
    
    print("="*40)
    print(f"Total Matched Tasks: {match_count}")
    print(f"Average Overlap Rate (Prefix-based): {avg_overlap_rate:.2%}")
    print("="*40)

    return results

parser = argparse.ArgumentParser(description="Verify maze solutions from a JSON file.")
parser.add_argument("--level", type=int, default=16, help="Level of the maze")
args = parser.parse_args()
LEVEL = args.level
BASE_DIR = f"/path/to/the/Maze/{LEVEL}_test"
TEST_DIR = f"{BASE_DIR}/result"

calculate_overlap_rate(os.path.join(BASE_DIR, "path.json"), os.path.join(TEST_DIR, "0_result.json"))