NAME="TSP" #Maze,Sudoku,TSP,FrozenLake,Edit
Model_Path="/path/to/the/mllm_ckpts/Qwen-Image-Edit-2511"
Data_Path="/path/to/the/${NAME}/data/dir/${NAME}" # example: xxx/DiffSynth-Studio/${NAME}
Output_Path="/path/to/the/${NAME}/save/dir/${NAME}_stage1" 

accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path "${Data_Path}" \
  --dataset_metadata_path "${Data_Path}/metadata_edit.csv" \
  --data_file_keys "image,edit_image,start_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --tokenizer_path "${Model_Path}/tokenizer"\
  --processor_path "${Model_Path}/processor"\
  --model_paths "[
      [
          \"${Model_Path}/transformer/diffusion_pytorch_model-00001-of-00005.safetensors\",
          \"${Model_Path}/transformer/diffusion_pytorch_model-00002-of-00005.safetensors\",
          \"${Model_Path}/transformer/diffusion_pytorch_model-00003-of-00005.safetensors\",
          \"${Model_Path}/transformer/diffusion_pytorch_model-00004-of-00005.safetensors\",
          \"${Model_Path}/transformer/diffusion_pytorch_model-00005-of-00005.safetensors\"
      ],
      [
          \"${Model_Path}/text_encoder/model-00001-of-00004.safetensors\",
          \"${Model_Path}/text_encoder/model-00002-of-00004.safetensors\",
          \"${Model_Path}/text_encoder/model-00003-of-00004.safetensors\",
          \"${Model_Path}/text_encoder/model-00004-of-00004.safetensors\"
      ],
      \"${Model_Path}/vae/diffusion_pytorch_model.safetensors\"
  ]" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${Output_Path}" \
  --lora_base_model "dit,text_encoder" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1;q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --lora_rank 32 \
  --find_unused_parameters \
  --zero_cond_t \
  --dataset_num_workers 8 \
  --save_steps 2000 \