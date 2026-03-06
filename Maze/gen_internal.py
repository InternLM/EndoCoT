# python gen_internal.py --size 8 --num 5000 --out ./8_train
import argparse
import random
import json
import os
import multiprocessing
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def make_empty_grid(n):
    return [[{"N": True, "E": True, "S": True, "W": True, "visited": False}
             for _ in range(n)] for _ in range(n)]

def neighbors_of(r, c, n):
    res = []
    if r > 0:    res.append(("N", r-1, c))
    if c < n-1:  res.append(("E", r, c+1))
    if r < n-1:  res.append(("S", r+1, c))
    if c > 0:    res.append(("W", r, c-1))
    return res

def remove_wall(grid, r, c, d):
    if d == "N": grid[r][c]["N"], grid[r-1][c]["S"] = False, False
    elif d == "S": grid[r][c]["S"], grid[r+1][c]["N"] = False, False
    elif d == "E": grid[r][c]["E"], grid[r][c+1]["W"] = False, False
    elif d == "W": grid[r][c]["W"], grid[r][c-1]["E"] = False, False

def gen_maze_dfs(n, rng=random):
    grid = make_empty_grid(n)
    stack = []
    sr, sc = rng.randrange(n), rng.randrange(n)
    grid[sr][sc]["visited"] = True
    stack.append((sr, sc))
    while stack:
        r, c = stack[-1]
        neighbors = [(d, nr, nc) for (d, nr, nc) in neighbors_of(r, c, n)
                     if not grid[nr][nc]["visited"]]
        if neighbors:
            d, nr, nc = rng.choice(neighbors)
            remove_wall(grid, r, c, d)
            grid[nr][nc]["visited"] = True
            stack.append((nr, nc))
        else:
            stack.pop()
    for row in grid:
        for cell in row:
            cell.pop("visited", None)
    return grid

def shortest_path_bfs(grid, start, end):
    n = len(grid)
    q = deque([(start, [start])])
    visited = {start}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == end:
            # 修复点：返回纯 Pythontuple 列表，避免 NumPy 类型报错
            return [tuple(map(int, p)) for p in path]
        cell = grid[r][c]
        for dr, dc, wall_self in [
            (-1, 0, "N"), (1, 0, "S"), (0, -1, "W"), (0, 1, "E")]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not cell[wall_self] and (nr, nc) not in visited:
                visited.add((nr, nc))
                new_path = list(path)
                new_path.append((nr, nc))
                q.append(((nr, nc), new_path))
    raise ValueError("No path exists")

def convert_path_to_udrl(path):
    if len(path) < 2: return ""
    moves = []
    for i in range(len(path) - 1):
        r1, c1 = path[i]; r2, c2 = path[i+1]
        if r2 < r1: moves.append('U')
        elif r2 > r1: moves.append('D')
        elif c2 < c1: moves.append('L')
        elif c2 > c1: moves.append('R')
    return "".join(moves)

def render_maze(grid, start, end, path=None, current_pos=None, size_px=512):
    n = len(grid)
    img = Image.new("RGB", (size_px, size_px), "black")
    draw = ImageDraw.Draw(img)
    cell_size_f = float(size_px) / n
    wall_w_f = cell_size_f / 4.0
    half_wall_f = wall_w_f / 2.0
    grid_w = max(1, int(cell_size_f / 16.0))

    for r in range(n):
        for c in range(n):
            x1, y1 = c * cell_size_f + half_wall_f, r * cell_size_f + half_wall_f
            x2, y2 = (c + 1) * cell_size_f - half_wall_f, (r + 1) * cell_size_f - half_wall_f
            draw.rectangle([(x1, y1), (x2, y2)], fill="white")
            cell = grid[r][c]
            if not cell["S"] and r < n - 1:
                draw.rectangle([(x1, y2), (x2, y2 + wall_w_f)], fill="white")
            if not cell["E"] and c < n - 1:
                draw.rectangle([(x2, y1), (x2 + wall_w_f, y2)], fill="white")

    def draw_dot(rc, color):
        r, c = rc
        cx, cy = c * cell_size_f + cell_size_f / 2, r * cell_size_f + cell_size_f / 2
        rad = max(2, int((cell_size_f - wall_w_f) * 0.25))
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=color)

    draw_dot(start, "yellow")
    draw_dot(end, "blue")

    if path is not None:
        pts = [(c * cell_size_f + cell_size_f / 2, r * cell_size_f + cell_size_f / 2) for r, c in path]
        draw.line(pts, fill="red", width=max(1, int(wall_w_f)), joint="curve")
    
    if current_pos is not None:
        draw_dot(current_pos, "green")

    return img

    def draw_dot(rc, color):
        r, c = rc
        cx, cy = c * cell_size_f + cell_size_f / 2, r * cell_size_f + cell_size_f / 2
        rad = max(2, int((cell_size_f - wall_w_f) * 0.25))
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=color)

    if current_pos is None:
        draw_dot(start, "yellow")
        draw_dot(end, "blue")
        if path is not None:
            pts = [(c * cell_size_f + cell_size_f / 2, r * cell_size_f + cell_size_f / 2) for r, c in path]
            draw.line(pts, fill="red", width=max(1, int(wall_w_f)), joint="curve")
    
    else:
        if path is not None:
            pts = [(c * cell_size_f + cell_size_f / 2, r * cell_size_f + cell_size_f / 2) for r, c in path]
            draw.line(pts, fill="red", width=max(1, int(wall_w_f)), joint="curve")
        
        draw_dot(start, "yellow")
        draw_dot(current_pos, "blue") 

    return img

def save_maze_to_text(grid, start, end, filepath):
    n = len(grid)
    with open(filepath, 'w') as f:
        f.write(f"{n}\n{start[0]} {start[1]}\n{end[0]} {end[1]}\n")
        for r in range(n):
            row_values = []
            for c in range(n):
                cell, value = grid[r][c], 0
                if cell["N"]: value |= 1
                if cell["S"]: value |= 2
                if cell["W"]: value |= 4
                if cell["E"]: value |= 8
                row_values.append(str(value))
            f.write(" ".join(row_values) + "\n")

def process_one_maze(args_tuple):
    i, size, min_len, out_dir = args_tuple
    rng = random.Random()
    
    grid, start, end, full_path = None, None, None, None
    for _ in range(500):
        g = gen_maze_dfs(size, rng)
        nodes = [(r, c) for r in range(size) for c in range(size)]
        s, e = rng.sample(nodes, 2)
        try:
            p = shortest_path_bfs(g, s, e)
            if len(p) >= min_len:
                grid, start, end, full_path = g, s, e, p
                break
        except ValueError: continue
    
    if grid is None: return None

    maze_id = f"{size}_{min_len}_{i:04d}"
    maze_folder = out_dir / maze_id
    final_folder = maze_folder / "final"
    steps_folder = maze_folder / "steps"
    
    for d in [maze_folder, final_folder, steps_folder]:
        d.mkdir(parents=True, exist_ok=True)
    
    render_maze(grid, start, end).save(maze_folder / f"{maze_id}.png")
    save_maze_to_text(grid, start, end, maze_folder / f"{maze_id}.txt")
    
    render_maze(grid, start, end, path=full_path).save(final_folder / "solution.png")
    
    intermediate_data = {}
    for j in range(0, len(full_path), 2):
        path_seg = full_path[:j+1]
        curr_pos = tuple(map(int, path_seg[-1]))
        
        step_name = f"step_{j:03d}"
        f_step_img = steps_folder / f"{step_name}.png"
        render_maze(grid, start, end, path=path_seg, current_pos=curr_pos).save(f_step_img)
        
        next_move = ""
        if j + 1 < len(full_path):
            next_move = convert_path_to_udrl(full_path[j:j+2])
            
        intermediate_data[f"{maze_id}_{step_name}"] = {
            "maze_id": maze_id,
            "step": j,
            "current_pos": curr_pos,
            "next_move": next_move,
            "img_rel_path": str(f_step_img.relative_to(out_dir))
        }
    
    return (maze_id, convert_path_to_udrl(full_path), intermediate_data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--num", type=int, default=40000, help="每个 min_len 生成的数量")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max_workers", type=int, default=64)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    
    min_len_list = [1, 6, 11, 16, 21]
    
    all_tasks = []
    for ml in min_len_list:
        for i in range(1, args.num + 1):
            all_tasks.append((i, args.size, ml, out_path))
            
    num_workers = 30
    print(f"开始生成任务：Size={args.size}, 总迷宫数={len(all_tasks)}, 进程数={num_workers}")
    
    final_paths_json = {}
    intermediate_paths_json = {}
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_one_maze, all_tasks), total=len(all_tasks), desc="批量生成中"))
        
    for res in results:
        if res:
            m_id, full_udrl, inter_dict = res
            final_paths_json[m_id] = full_udrl
            intermediate_paths_json.update(inter_dict)
            
    with open(out_path / "path_final.json", "w") as f:
        json.dump(dict(sorted(final_paths_json.items())), f, indent=4)
        
    with open(out_path / "path_intermediate.json", "w") as f:
        json.dump(dict(sorted(intermediate_paths_json.items())), f, indent=4)

    print(f"\n生成完毕！文件保存在: {out_path.resolve()}")

if __name__ == "__main__":
    main()