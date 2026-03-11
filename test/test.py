import torch
import os
import argparse
from safetensors.torch import load_file
from diffusers.utils import load_image
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

def parse_args():
    parser = argparse.ArgumentParser(description="EndoCoT Inference Script")
    parser.add_argument("--model_root", type=str, default="/path/to/EndoCoT/ckpts_fix", help="Root directory of the merged checkpoints")
    
    parser.add_argument("--lora_paths", type=str, nargs='+', required=True, help="List of LoRA .safetensors files")
    
    parser.add_argument("--task", type=str, default="Edit", choices=["Sudoku", "FrozenLake", "TSP", "Maze"], help="Task type")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save output images")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    return parser.parse_args()

def get_task_prompt(task):
    prompts = {
        "Sudoku": "Solve this Sudoku puzzle step-by-step from top-left to bottom-right. Identify the empty cell, and fill in the correct digit.",
        "FrozenLake": "Draw a continuous red line connecting the Start point to the Goal point step-by-step, avoiding all holes. Mark the current end of the path with a green dot. The line must be drawn through the center of the grid cells rather than along the grid lines.",
        "TSP": "Draw a continuous red line step-by-step from the start point to form the shortest closed loop, marking the current endpoint with a green dot. Avoiding another circle.",
        "Maze": "Draw a continuous red line starting from the yellow dot towards the blue dot step-by-step, avoiding all walls. Mark the current end of the path with a green dot.",
    }
    return prompts.get(task, "")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        "transformer": os.path.join(args.model_root, "transformer/diffusion_pytorch_model_merged.safetensors"),
        "text_encoder": os.path.join(args.model_root, "text_encoder/model_merged.safetensors"),
        "vae": os.path.join(args.model_root, "vae/diffusion_pytorch_model.safetensors"),
        "processor": os.path.join(args.model_root, "processor/"),
        "tokenizer": os.path.join(args.model_root, "tokenizer/")
    }

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=config["transformer"]),
            ModelConfig(path=config["text_encoder"]),
            ModelConfig(path=config["vae"]),
        ],
        processor_config=ModelConfig(path=config["processor"]),
        tokenizer_config=ModelConfig(path=config["tokenizer"]),
    )

    for lora in args.lora_paths:
        if os.path.exists(lora):
            print(f"Loading LoRA: {lora}")
            pipe.load_lora(pipe.text_encoder, lora, alpha=1.0)
            pipe.load_lora(pipe.dit, lora, alpha=1.0)
        else:
            print(f"Warning: LoRA file {lora} not found, skipping.")

    prompt = get_task_prompt(args.task)
    input_img = load_image(args.input_image)
    
    generated_image, intermediate_images = pipe(
        prompt, 
        edit_image=input_img, 
        seed=args.seed, 
        num_inference_steps=args.steps, 
        height=512, 
        width=512,    
        edit_image_auto_resize=False,
        zero_cond_t=True,
        return_intermediates=True,
    )

    generated_image.save(os.path.join(args.output_dir, "result.png"))
    input_img.save(os.path.join(args.output_dir, "original.png"))

    intermediate_dir = os.path.join(args.output_dir, "reasoning_steps")
    os.makedirs(intermediate_dir, exist_ok=True)
    for i, img in enumerate(intermediate_images):
        img.save(os.path.join(intermediate_dir, f"step_{i:02d}.png"))
    
if __name__ == "__main__":
    main()