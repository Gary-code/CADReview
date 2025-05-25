bash ./my_experiment/vllm_deploy.sh \
    --cuda_devices "0,1,2,3" \
    --model_path "Llama-3___2-11B-Vision-Instruct" \
    --template "llama3_2_vision" \
    --inference_backend "vllm" \
    --enforce_eager true \
    --max_len 6144 \
    --max_num_seqs 16 \
    --lora_ckpt "./outputs/code_editor_directly_llama/v0-20250107-180648/checkpoint-542"
