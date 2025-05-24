bash ./my_experiment/vllm_deploy.sh \
    --cuda_devices "0,1,2,3,4,5,6,7" \
    --model_path "./outputs/alignment_code_completion/qwen2_vl_grounding_init/" \
    --template "qwen2_vl" \
    --inference_backend "vllm" \
    --enforce_eager true \
    --max_len 6144 \
    --lora_ckpt "./outputs/code_editor_code_completion_init/"
