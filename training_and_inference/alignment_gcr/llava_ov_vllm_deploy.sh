bash ./vllm_deploy.sh \
    --cuda_devices "0,1,2,3,4,5,6,7" \
    --model_path "llava-onevision-qwen2-7b-ov-hf" \
    --template "llava_onevision_hf" \
    --inference_backend "vllm" \
    --enforce_eager true \
    --max_len 6144 \
    --lora_ckpt "./outputs/alignment_grounding/llava_ov/"
