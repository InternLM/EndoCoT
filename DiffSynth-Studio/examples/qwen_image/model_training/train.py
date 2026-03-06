import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core.data.operators import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

import torch._dynamo
torch._dynamo.disable()

# # dxl: 会慢很多
# torch.autograd.set_detect_anomaly(True)

class QwenImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        edit_image_auto_resize = False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        zero_cond_t=False,
        text_encoder=None,
    ):
        super().__init__()

        # model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        # tokenizer_config = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        # processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/") if processor_path is None else ModelConfig(processor_path)
        # self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, processor_config=processor_config)




        ROOT_DIR = f"/mnt/shared-storage-user/mllm/shared/mllm_ckpts/DiffThinker/Qwen-Image-Edit-2511"
        # dxl：Maze
        # LORA_MODEL_PATH = "/mnt/shared-storage-user/mllmexp/daixuanlang/code/DiffThinker_baseline/DiffThinker-main/eos2/step-56000-fix.safetensors"
        # # dxl: Sudoku
        # # LORA_MODEL_PATH = "/mnt/shared-storage-user/mllmexp/daixuanlang/code/DiffThinker_baseline/DiffThinker-main/Sudoku_stage1/step-50000-fix.safetensors"
        # LORA_MODEL_PATH = "/mnt/shared-storage-user/mllm/daixuanlang/code/Diffthinker/Sudoku_stage1_40000/step-118000-fix.safetensors"
        # # dxl: FrozenLake
        # LORA_MODEL_PATH = "/mnt/shared-storage-user/mllmexp/daixuanlang/code/DiffThinker_baseline/DiffThinker-main/Frozen_Lake_stage1/step-72000-fix.safetensors"
        # # dxl: TSP
        # # LORA_MODEL_PATH = f"/mnt/shared-storage-user/mllmexp/daixuanlang/code/DiffThinker_baseline/DiffThinker-main/TSP_stage1/step-82000-fix.safetensors"
        # # LORA_MODEL_PATH = "/mnt/shared-storage-user/mllm/daixuanlang/code/Diffthinker/TSP_stage1_30000/step-66000-fix.safetensors"
        # LORA_MODEL_PATH = "/mnt/shared-storage-user/mllm/daixuanlang/code/Diffthinker/TSP_stage1_30000/step-200000-fix.safetensors"
        # # dxl: Onlydit
        # LORA_MODEL_PATH = "/mnt/shared-storage-user/mllmexp/daixuanlang/code/DiffThinker_baseline/DiffThinker-main/onlydit_maze/step-56000-fix.safetensors"
        # # dxl: all
        # # LORA_MODEL_PATH = "/mnt/shared-storage-user/mllm/daixuanlang/code/Diffthinker/all_stage1_avg/step-90000-fix.safetensors"
        # LORA_MODEL_PATH1 = "/mnt/shared-storage-user/mllm/daixuanlang/code/Diffthinker/all_stage1_avg/step-400000-fix.safetensors"
        # LORA_MODEL_PATH2 = "/mnt/shared-storage-user/mllm/daixuanlang/code/Diffthinker/all_stage2_avg_400000/step-2000-fix.safetensors"
        # LORA_MODEL_PATH3 = "/mnt/shared-storage-user/mllm/daixuanlang/code/Diffthinker/all_stage2_sudoku_stage3/step-800-fix.safetensors"
        # dxl: Edit
        LORA_MODEL_PATH1 = "/mnt/shared-storage-user/mllm/daixuanlang/code/Diffthinker/image_edit_4/step-104000-fix.safetensors"

        transformer_path = os.path.join(ROOT_DIR, "transformer", "diffusion_pytorch_model_merged.safetensors")
        text_encoder_path = os.path.join(ROOT_DIR, "text_encoder", "model_merged.safetensors")
        vae_path = os.path.join(ROOT_DIR, "vae", "diffusion_pytorch_model.safetensors")
        processor_path = os.path.join(ROOT_DIR, "processor/")
        tokenizer_path = os.path.join(ROOT_DIR, "tokenizer/")

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(path=transformer_path),
                ModelConfig(path=text_encoder_path),
                ModelConfig(path=vae_path),
            ],
            processor_config=ModelConfig(path=processor_path),
            tokenizer_config=ModelConfig(path=tokenizer_path),
        )
        self.pipe.load_lora(self.pipe.text_encoder, LORA_MODEL_PATH1, alpha=1)
        self.pipe.load_lora(self.pipe.dit, LORA_MODEL_PATH1, alpha=1)
        # self.pipe.load_lora(self.pipe.text_encoder, LORA_MODEL_PATH2, alpha=1)
        # self.pipe.load_lora(self.pipe.dit, LORA_MODEL_PATH2, alpha=1)
        # self.pipe.load_lora(self.pipe.text_encoder, LORA_MODEL_PATH3, alpha=1)
        # self.pipe.load_lora(self.pipe.dit, LORA_MODEL_PATH3, alpha=1)





        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # dxl：用来存之前的prompt的
        self.prev_prompt = ""
        self.idx = 1
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # dxl: 可以看一下哪些参数可以训练
        for name, sub_model in [("Transformer", self.pipe.dit), ("Text Encoder", self.pipe.text_encoder), ("VAE", self.pipe.vae), ("MLP", self.pipe.mlp)]:
            trainable_params = sum(p.numel() for p in sub_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in sub_model.parameters())
            print(f"{name}: {'Trainable' if trainable_params > 0 else 'Frozen'} ({trainable_params}/{total_params} params)")
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.edit_image_auto_resize = edit_image_auto_resize
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.zero_cond_t = zero_cond_t
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
            "zero_cond_t": self.zero_cond_t,
            "edit_image_auto_resize": self.edit_image_auto_resize,
        }
        # Assume you are using this pipeline for inference,
        # please fill in the input parameters.
        if isinstance(data["image"], list):
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"][0].size[1],
                "width": data["image"][0].size[0],
            })
        else:
            inputs_shared.update({
                "input_image": data["image"],
                "height": data["image"].size[1],
                "width": data["image"].size[0],
            })
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    # def forward(self, data, inputs=None):
    #     if inputs is None: inputs = self.get_pipeline_inputs(data)
    #     inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
    #     for unit in self.pipe.units:
    #         inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
    #     loss = self.task_to_loss[self.task](self.pipe, *inputs)
    #     inputs[0]['input_image'].save('/mnt/shared-storage-user/mllmexp/daixuanlang/code/DiffThinker_baseline/DiffThinker-main/test/input_debug.png')
    #     inputs[0]['edit_image'].save('/mnt/shared-storage-user/mllmexp/daixuanlang/code/DiffThinker_baseline/DiffThinker-main/test/edit_debug.png')
    #     print(f"目前的loss是：{loss}")
    #     breakpoint()
    #     return loss

    def forward(self, data, inputs=None):
        eos_loss = None
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        # dxl: 这里修改了一下
        if data['start_image']:
            inputs[1]["prompt"] = data['prompt']
        else:
            try:
                if data['idx']!=None:
                    inputs[1]["prompt"] = data['prompt']
            except:
                inputs[1]["prompt"] = self.prev_prompt

        # # dxl：对于stage2，不用保存prev prompt
        # inputs[1]["prompt"] = data['prompt']

        # dxl: version 2 来判断是不是final image
        inputs[1]['is_final'] = data['final_image']
        try:
            inputs[1]['gt_prompt'] = data['gt_prompt']
        except:
            inputs[1]['gt_prompt'] = None
        try:
            inputs[1]['idx'] = data['idx']
        except:
            inputs[1]['idx'] = None

        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        # dxl：version2\4， 如果要引入eos的话这里要加个ce loss
        eos_loss = inputs[1]["eos_loss"]

        self.prev_prompt = inputs[1]["prompt"].detach()
        loss = self.task_to_loss[self.task](self.pipe, *inputs)

        # dxl: version3， 如果把final状态的loss给scale一下
        # if data['final_image']:
        #     loss *= self.idx
        #     print(f"Current index is:{self.idx}")
        #     self.idx = 1
        # else:
        #     self.idx += 1

        # inputs[0]['input_image'].save('/mnt/shared-storage-user/mllmexp/daixuanlang/code/DiffThinker_baseline/DiffThinker-main/test/input_debug.png')
        # inputs[0]['edit_image'].save('/mnt/shared-storage-user/mllmexp/daixuanlang/code/DiffThinker_baseline/DiffThinker-main/test/edit_debug.png')
        # print(f"目前的prompt shape是：{inputs[1]['prompt'].shape}")
        print(f"目前的loss是：{loss}")
        # breakpoint()

        # print(f"目前的image path是：{data['image_path']}")

        if eos_loss is not None:
            print(f"目前的eos loss是：{eos_loss}")
            return loss + eos_loss
        else:
            return loss


def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--text_encoder_path", type=str, default=None, help="Path to the text encoder. If provided, the processor will be used for image editing.")
    parser.add_argument("--zero_cond_t", default=False, action="store_true", help="A special parameter introduced by Qwen-Image-Edit-2511. Please enable it for this model.")
    parser.add_argument("--edit_image_auto_resize", default=False, action="store_true")
    return parser


if __name__ == "__main__":
    parser = qwen_image_parser()
    args = parser.parse_args()
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
        special_operator_map={
            # Qwen-Image-Layered
            "layer_input_image": ToAbsolutePath(args.dataset_base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16),
            "image": RouteByType(operator_map=[
                (str, ToAbsolutePath(args.dataset_base_path) >> LoadImage() >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16)),
                (list, SequencialProcess(ToAbsolutePath(args.dataset_base_path) >> LoadImage(convert_RGB=False, convert_RGBA=True) >> ImageCropAndResize(args.height, args.width, args.max_pixels, 16, 16))),
            ])
        }
    )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    model = QwenImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        edit_image_auto_resize = args.edit_image_auto_resize,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
        # device="cpu",
        zero_cond_t=args.zero_cond_t,
        text_encoder=args.text_encoder_path
    )
    # model.to(accelerator.device)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    # breakpoint()
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
