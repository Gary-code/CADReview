bash ./my_experiment/vllm_deploy.sh \
    --cuda_devices "0,1,2,3" \
    --model_path "./outputs/alignment_code_completion/llava_ov_grounding_init/" \
    --template "llava_onevision_hf" \
    --inference_backend "vllm" \
    --enforce_eager true \
    --max_len 12000 \
    --lora_ckpt "./outputs/code_editor_code_completion_init/llava-ov/"
