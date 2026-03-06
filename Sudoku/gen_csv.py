import os
import json
import pandas as pd

def process_single_directory(root_dir, is_final):
    data = []
    
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    path_json = os.path.join(root_dir, "path_final.json")
    if not os.path.exists(path_json):
        print(f"Warning: {path_json} not found. Skipping...")
        return []

    with open(path_json, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    prefix_prompt = "Solve this Sudoku puzzle step-by-step from top-left to bottom-right. Identify the empty cell, and fill in the correct digit."

    for folder_name in sorted(subdirs):
        if folder_name not in json_data:
            continue
            
        reasoning_path = json_data[folder_name]
        folder_path = os.path.join(root_dir, folder_name)
        
        current_image_ref = os.path.join(root_dir, folder_name, f"problem.png")
        
        steps_dir = os.path.join(folder_path, "steps")
        idx = 0
        if os.path.exists(steps_dir):
            step_files = sorted([f for f in os.listdir(steps_dir) if f.endswith('.png')])
            for step_file in step_files:
                if not is_final:
                    data.append({
                        "edit_image": current_image_ref,
                        "start_image": idx==0,
                        "final_image": False,
                        "image": os.path.join(root_dir, folder_name, "steps", step_file),
                        "prompt": prefix_prompt,
                        # "gt_prompt": prefix_prompt + " " + reasoning_path[:idx]
                    })
                idx += 2

        # 2. 处理 final 文件夹
        final_dir = os.path.join(folder_path, "final")
        solution_path = os.path.join(final_dir, "solution.png")
        if os.path.exists(solution_path):
            if not is_final:
                data.append({
                    "edit_image": current_image_ref,
                    "start_image": False,
                    "final_image": True,
                    "image": solution_path,
                    "prompt": prefix_prompt,
                    "gt_prompt": prefix_prompt + " " + reasoning_path,
                    # "idx": idx
                })
            else:
                data.append({
                    "edit_image": current_image_ref,
                    "start_image": False,
                    "final_image": True,
                    "image": solution_path,
                    "prompt": prefix_prompt,
                    "gt_prompt": prefix_prompt + " " + reasoning_path,
                    "idx": idx
                })
            idx += 2
            
    return data

def main():
    # 需要处理的三个目标目录
    target_directories = [
        "30_train",
        "35_train",
        "40_train",
        "45_train",
        "30_train+",
        "35_train+",
        "40_train+",
        "45_train+",
    ]

    all_data = []

    for directory in target_directories:
        print(f"Processing: {directory}")
        if os.path.exists(directory):
            folder_data = process_single_directory(directory, False)
            all_data.extend(folder_data)
        else:
            print(f"Directory not found: {directory}")

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv("metadata_edit.csv", index=False)
        print(f"Success! Total rows saved: {len(df)}")
    else:
        print("No data collected.")

    all_data = []

    for directory in target_directories:
        print(f"Processing: {directory}")
        if os.path.exists(directory):
            folder_data = process_single_directory(directory, True)
            all_data.extend(folder_data)
        else:
            print(f"Directory not found: {directory}")

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv("metadata_edit_final.csv", index=False)
        print(f"Success! Total rows saved: {len(df)}")
    else:
        print("No data collected.")

        
if __name__ == "__main__":
    main()