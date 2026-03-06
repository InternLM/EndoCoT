import argparse
import random
import json
import os
import multiprocessing
import math
import numpy as np
from pathlib import Path
from python_tsp.exact import solve_tsp_dynamic_programming
from PIL import Image, ImageDraw
from tqdm import tqdm

def solve_tsp_optimal(points):
    num_points = len(points)
    dist_matrix = np.zeros((num_points, num_points))
    
    for i in range(num_points):
        for j in range(num_points):
            dist = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            dist_matrix[i][j] = dist
            
    permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
    optimal_path = [points[i] for i in permutation]
    optimal_path.append(optimal_path[0])
    return optimal_path

def generate_one_tsp_data(grid_size, num_points):
    valid_range = range(0, grid_size)
    candidates = [(x, y) for x in valid_range for y in valid_range]
    if len(candidates) < num_points:
        raise ValueError(f"{grid_size}x{grid_size} too small")
    points = random.sample(candidates, num_points)
    solution_path = solve_tsp_optimal(points)
    return points, solution_path

def render_tsp(grid_size, points, path=None, current_node=None, size_px=512):
    img = Image.new("RGB", (size_px, size_px), "white")
    draw = ImageDraw.Draw(img)
    cell_size = size_px / grid_size
    
    def to_px(p):
        return (p[0] * cell_size + cell_size / 2, 
                p[1] * cell_size + cell_size / 2)

    grid_color = (240, 240, 240) 
    line_width = max(1, int(size_px / grid_size / 20))
    for i in range(1, grid_size):
        pos = i * cell_size
        draw.line([(pos, 0), (pos, size_px)], fill=grid_color, width=line_width)
        draw.line([(0, pos), (size_px, pos)], fill=grid_color, width=line_width)

    if path and len(path) > 1:
        px_path = [to_px(p) for p in path]
        path_width = max(2, int(cell_size * 0.15))
        draw.line(px_path, fill="red", width=path_width, joint="curve")

    radius = max(3, int(cell_size * 0.3))
    start_node = points[0]
    for p in points:
        px, py = to_px(p)
        if p == start_node:
            color = "yellow"
        elif p == current_node:
            color = "green"
        else:
            color = "blue"
        
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], 
                     fill=color, outline="black", width=1)

    return img

def path_to_string(path):
    return "->".join([f"({p[0]},{p[1]})" for p in path])

def process_one_tsp(args_tuple):
    i, size, num_points, out_dir = args_tuple
    
    try:
        points, full_path = generate_one_tsp_data(size, num_points)
    except Exception:
        return None

    tsp_id = f"{size}_{num_points}_{i:04d}"
    tsp_folder = out_dir / tsp_id
    final_folder = tsp_folder / "final"
    steps_folder = tsp_folder / "steps"
    
    for d in [tsp_folder, final_folder, steps_folder]:
        d.mkdir(parents=True, exist_ok=True)

    render_tsp(size, points).save(tsp_folder / f"{tsp_id}.png")
    
    render_tsp(size, points, path=full_path).save(final_folder / "solution.png")
    
    intermediate_data = {}
    for j in range(len(full_path)):
        path_so_far = full_path[:j+1]
        curr_pos = tuple(map(int, full_path[j]))
        
        step_name = f"step_{j:03d}"
        f_step_img = steps_folder / f"{step_name}.png"
        
        render_tsp(size, points, path=path_so_far, current_node=curr_pos).save(f_step_img)
        
        next_node = ""
        if j + 1 < len(full_path):
            next_node = f"({full_path[j+1][0]},{full_path[j+1][1]})"
            
        intermediate_data[f"{tsp_id}_{step_name}"] = {
            "tsp_id": tsp_id,
            "step": j,
            "current_pos": curr_pos,
            "next_node": next_node,
            "img_rel_path": str(f_step_img.relative_to(out_dir))
        }
    
    return (tsp_id, path_to_string(full_path), intermediate_data)

def main():
    p = argparse.ArgumentParser(description="Generate TSP solution steps.")
    p.add_argument("--size", type=int, required=True, help="Grid size")
    p.add_argument("--num", type=int, default=10, help="Number of TSP instances")
    p.add_argument("--points", type=int, default=5, help="Number of points per TSP")
    p.add_argument("--out", type=str, required=True, help="Output directory")
    p.add_argument("--max_workers", type=int, default=30)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    tasks = [(i, args.size, args.points, out_path) for i in range(1, args.num + 1)]
    
    num_workers = 30
    print(f"Generating {args.num} TSP tasks: {args.size}x{args.size}, {args.points} Points")
    
    final_paths_json = {}
    intermediate_paths_json = {}
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_one_tsp, tasks), total=len(tasks), desc="Processing"))
        
    for res in results:
        if res:
            t_id, full_path_str, inter_dict = res
            final_paths_json[t_id] = full_path_str
            intermediate_paths_json.update(inter_dict)
            
    with open(out_path / "path_final.json", "w", encoding="utf-8") as f:
        json.dump(dict(sorted(final_paths_json.items())), f, indent=4)
        
    with open(out_path / "path_intermediate.json", "w", encoding="utf-8") as f:
        json.dump(dict(sorted(intermediate_paths_json.items())), f, indent=4)

    print(f"\nDone! Results saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()