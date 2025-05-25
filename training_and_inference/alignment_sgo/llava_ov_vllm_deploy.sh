bash ./my_experiment/vllm_deploy.sh \
    --cuda_devices "0,1,2,3" \
    --model_path "llava-onevision-qwen2-7b-ov-hf" \
    --template "llava_onevision_hf" \
    --inference_backend "vllm" \
    --enforce_eager true \
    --max_len 6144 \
    --lora_ckpt "./outputs/alignment_code_completion/llava_ov/"
