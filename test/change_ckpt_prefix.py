import argparse
from safetensors.torch import load_file, save_file

def parse_args():
    parser = argparse.ArgumentParser(description="Fix LoRA key names for EndoCoT compatibility.")
    parser.add_argument("--src", type=str, required=True, help="Path to the original LoRA .safetensors file.")
    parser.add_argument("--dst", type=str, default=None, help="Path to save the fixed LoRA file. If not set, will append '-fix' to src.")
    
    args = parser.parse_args()
    
    if args.dst is None:
        args.dst = args.src.replace(".safetensors", "-fix.safetensors")
        
    return args

def main():
    args = parse_args()

    print(f"Loading LoRA from: {args.src}")
    state = load_file(args.src)

    fixed_state = {}
    prefixes_to_remove = ["pipe.text_encoder.", "pipe.mlp."]

    for k, v in state.items():
        original_key = k
        for prefix in prefixes_to_remove:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        
        fixed_state[k] = v

    save_file(fixed_state, args.dst)

    print("-" * 30)
    print(f"Successfully processed {len(fixed_state)} keys.")
    print(f"Saved fixed LoRA to:\n{args.dst}")

if __name__ == "__main__":
    main()