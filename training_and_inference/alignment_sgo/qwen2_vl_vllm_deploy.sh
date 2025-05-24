bash ./my_experiment/vllm_deploy.sh \
    --cuda_devices "0,1,2,3,4,5,6,7" \
    --model_path "qwen2-vl-7b-instruct" \
    --template "qwen2_vl" \
    --inference_backend "vllm" \
    --enforce_eager true \
    --max_len 6144 \
    --lora_ckpt "./outputs/alignment_code_completion/qwen2_vl/"

