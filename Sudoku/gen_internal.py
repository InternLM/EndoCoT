import argparse
import random
import os
import multiprocessing
import copy
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import json

class SudokuGenerator:
    def __init__(self, size=9):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.solution = None
        self.solving_steps = []

    def _is_valid(self, grid, row, col, num):
        if num in grid[row]: return False
        if num in [grid[i][col] for i in range(self.size)]: return False
        sr, sc = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if grid[sr + i][sc + j] == num: return False
        return True

    def _get_empty(self, grid):
        for i in range(self.size):
            for j in range(self.size):
                if grid[i][j] == 0: return i, j
        return None

    def fill_grid(self):
        empty = self._get_empty(self.grid)
        if not empty: return True
        r, c = empty
        nums = list(range(1, 10))
        random.shuffle(nums)
        for n in nums:
            if self._is_valid(self.grid, r, c, n):
                self.grid[r][c] = n
                if self.fill_grid(): return True
                self.grid[r][c] = 0
        return False

    def count_solutions(self, grid):
        count = 0
        def solve(g):
            nonlocal count
            empty = self._get_empty(g)
            if not empty:
                count += 1
                return
            r, c = empty
            for n in range(1, 10):
                if self._is_valid(g, r, c, n):
                    g[r][c] = n
                    solve(g)
                    g[r][c] = 0
                    if count >= 2: return
        solve(copy.deepcopy(grid))
        return count

    def generate_puzzle(self, exist_count):
        self.fill_grid()
        self.solution = copy.deepcopy(self.grid)
        cells = list(range(81))
        random.shuffle(cells)
        removed = 0
        target_remove = 81 - exist_count
        for idx in cells:
            if removed >= target_remove: break
            r, c = idx // 9, idx % 9
            temp = self.grid[r][c]
            self.grid[r][c] = 0
            if self.count_solutions(self.grid) != 1:
                self.grid[r][c] = temp
            else:
                removed += 1
        return self.grid, self.solution

    def record_solve_steps(self, puzzle_grid):
        self.solving_steps = []
        current_grid = copy.deepcopy(puzzle_grid)
        
        def solver(g):
            empty = self._get_empty(g)
            if not empty: return True
            r, c = empty
            correct_val = self.solution[r][c]
            g[r][c] = correct_val
            self.solving_steps.append({
                "grid": copy.deepcopy(g),
                "cell": (r, c),
                "val": correct_val
            })
            if solver(g): return True
            return False

        solver(current_grid)
        return self.solving_steps

def render_sudoku(grid, out_path, puzzle_mask=None, highlight_cell=None, size_px=512):
    img = Image.new("RGB", (size_px, size_px), "white")
    draw = ImageDraw.Draw(img)
    cell_size = size_px / 9
    
    if highlight_cell:
        r, c = highlight_cell
        draw.rectangle([c*cell_size, r*cell_size, (c+1)*cell_size, (r+1)*cell_size], fill="#E6F4EA")

    for i in range(10):
        w = 3 if i % 3 == 0 else 1
        draw.line([(i*cell_size, 0), (i*cell_size, size_px)], fill="black", width=w)
        draw.line([(0, i*cell_size), (size_px, i*cell_size)], fill="black", width=w)

    try:
        font = ImageFont.truetype("/mnt/shared-storage-user/mllmexp/daixuanlang/code/DiffThinker_baseline/DiffThinker-main/Sudoku/思源黑体SourceHanSansCN-Light.otf", int(cell_size))
    except:
        font = ImageFont.load_default()

    for i in range(9):
        for j in range(9):
            val = grid[i][j]
            if val != 0:
                is_fixed = puzzle_mask and puzzle_mask[i][j] != 0
                color = "black" if is_fixed else "#1A73E8"
                if (i, j) == highlight_cell: color = "red"
                
                txt = str(val)
                bbox = draw.textbbox((0,0), txt, font=font)
                x = j * cell_size + (cell_size - (bbox[2]-bbox[0]))/2
                y = i * cell_size + (cell_size - (bbox[3]-bbox[1]))/2
                draw.text((x, y), txt, fill=color, font=font)
    
    img.save(out_path)

def process_one_sudoku(args_tuple):
    idx, exist, out_dir = args_tuple
    gen = SudokuGenerator()
    puzzle, solution = gen.generate_puzzle(exist)
    puzzle_mask = copy.deepcopy(puzzle) 
    steps = gen.record_solve_steps(puzzle)
    
    sid = f"sudoku_{exist}_{idx:03d}"
    s_folder = out_dir / sid
    
    steps_folder = s_folder / "steps"
    final_folder = s_folder / "final"
    steps_folder.mkdir(parents=True, exist_ok=True)
    final_folder.mkdir(parents=True, exist_ok=True)

    render_sudoku(puzzle, s_folder / "problem.png", puzzle_mask=puzzle)
    
    render_sudoku(solution, final_folder / "solution.png", puzzle_mask=puzzle_mask)
    
    step_metadata = {}
    for i, step in enumerate(steps):
        step_name = f"step_{i:03d}"
        f_path = steps_folder / f"{step_name}.png"
        render_sudoku(step["grid"], f_path, puzzle_mask=puzzle_mask, highlight_cell=step["cell"])
        
        step_metadata[f"{sid}_{step_name}"] = {
            "cell": step["cell"],
            "val": step["val"],
            "rel_path": str(f_path.relative_to(out_dir))
        }
    return sid, "".join(map(str, [c for r in solution for c in r])), step_metadata

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=1)
    p.add_argument("--exist", type=int, default=30)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    tasks = [(i, args.exist, out) for i in range(1, args.num + 1)]
    final_json, inter_json = {}, {}

    print(f"Generating Sudoku Steps...")
    with multiprocessing.Pool(processes=64) as pool:
        results = list(tqdm(pool.imap_unordered(process_one_sudoku, tasks), total=args.num))

    for sid, sol_str, steps in results:
        final_json[sid] = sol_str
        inter_json.update(steps)

    with open(out / "path_final.json", "w") as f: json.dump(final_json, f, indent=4)
    with open(out / "path_intermediate.json", "w") as f: json.dump(inter_json, f, indent=4)
    print(f"Done! Saved to {out}")

if __name__ == "__main__":
    main()