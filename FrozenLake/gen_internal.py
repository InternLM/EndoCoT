# python your_script_name.py --size 8 --num 10 --out ./dataset_output
import os
import warnings
import argparse
import random
import gymnasium as gym
import networkx as nx
from PIL import Image, ImageDraw
import multiprocessing
from multiprocessing import Pool, Manager
from tqdm import tqdm
import json
from pathlib import Path

os.environ["SDL_AUDIODRIVER"] = "dummy"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()

def generate_random_layout(size, p=0.8):
    map_grid = [['' for _ in range(size)] for _ in range(size)]
    all_coords = [(r, c) for r in range(size) for c in range(size)]
    start_pos, goal_pos = random.sample(all_coords, 2)
    
    for r in range(size):
        for c in range(size):
            if (r, c) == start_pos:
                map_grid[r][c] = 'S'
            elif (r, c) == goal_pos:
                map_grid[r][c] = 'G'
            else:
                map_grid[r][c] = 'F' if random.random() < p else 'H'
    return ["".join(row) for row in map_grid]

def get_shortest_path(desc):
    rows = len(desc)
    cols = len(desc[0])
    G = nx.Graph()
    start_pos, goal_pos = None, None

    for r in range(rows):
        for c in range(cols):
            char = desc[r][c]
            if char == 'S': start_pos = (r, c)
            if char == 'G': goal_pos = (r, c)
            if char == 'H': continue
            
            G.add_node((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and desc[nr][nc] != 'H':
                    G.add_edge((r, c), (nr, nc))

    try:
        if start_pos and goal_pos:
            return nx.shortest_path(G, source=start_pos, target=goal_pos)
    except:
        return None
    return None

def draw_path(img, path, size, current_pos=None):
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cell = w / size

    def to_px(r, c):
        return (c * cell + cell / 2, r * cell + cell / 2)

    if path and len(path) > 1:
        px_path = [to_px(r, c) for r, c in path]
        draw.line(px_path, fill="red", width=max(2, int(cell * 0.2)), joint="curve")

    if current_pos:
        r, c = current_pos
        x, y = to_px(r, c)
        radius = int(cell * 0.3)
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill="green", outline="black")

    return img

def path_to_string(path):
    return "->".join([f"({r},{c})" for r,c in path])

def process_one_frozenlake(args_tuple):
    i, size, p, min_len, out_dir = args_tuple
    random.seed()

    while True:
        desc = generate_random_layout(size, p)
        path = get_shortest_path(desc)
        if path and len(path) - 1 >= min_len:
            break

    fl_id = f"{size}_{min_len}_{p}_{i:04d}"
    fl_folder = out_dir / fl_id
    final_folder = fl_folder / "final"
    steps_folder = fl_folder / "steps"

    for d in [fl_folder, final_folder, steps_folder]:
        d.mkdir(parents=True, exist_ok=True)

    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False, render_mode="rgb_array")
    env.reset()
    base_img = Image.fromarray(env.render()).resize((512, 512), Image.NEAREST)
    env.close()

    base_img.save(fl_folder / f"{fl_id}.png")

    final_img = base_img.copy()
    draw_path(final_img, path, size).save(final_folder / "solution.png")

    intermediate_data = {}
    STEP_STRIDE = 3
    step_indices = list(range(0, len(path), STEP_STRIDE))
    if step_indices[-1] != len(path) - 1:
        step_indices.append(len(path) - 1)

    for j in step_indices:
        path_so_far = path[:j+1]
        curr_pos = path[j]
        step_name = f"step_{j:03d}"
        step_img_path = steps_folder / f"{step_name}.png"

        step_img = base_img.copy()
        draw_path(step_img, path_so_far, size, current_pos=curr_pos)
        step_img.save(step_img_path)

        next_node = ""
        if j + 1 < len(path):
            next_node = f"({path[j+1][0]},{path[j+1][1]})"

        intermediate_data[f"{fl_id}_{step_name}"] = {
            "fl_id": fl_id,
            "step": j,
            "current_pos": curr_pos,
            "next_node": next_node,
            "img_rel_path": str(step_img_path.relative_to(out_dir))
        }

    return fl_id, path_to_string(path), intermediate_data

def load_json_if_exists(p):
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    tasks = [(i, args.size, args.p, args.min_len, out_path) for i in range(1, args.num+1)]

    final_paths_json = {}
    intermediate_paths_json = {}

    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_one_frozenlake, tasks), total=len(tasks)))

    for res in results:
        if res:
            fl_id, path_str, inter_dict = res
            final_paths_json[fl_id] = path_str
            intermediate_paths_json.update(inter_dict)

    final_json_path = out_path / "path_final.json"
    inter_json_path = out_path / "path_intermediate.json"

    old_final = load_json_if_exists(final_json_path)
    old_inter = load_json_if_exists(inter_json_path)

    old_final.update(final_paths_json)
    old_inter.update(intermediate_paths_json)

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(old_final.items())), f, indent=4)

    with open(inter_json_path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(old_inter.items())), f, indent=4)

    print("Done.")

if __name__ == "__main__":
    main()
