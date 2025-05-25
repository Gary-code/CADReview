bash ./my_experiment/vllm_deploy.sh \
    --cuda_devices "0,1,2,3,4,5,6,7" \
    --model_path "./outputs/alignment_grounding/" \
    --template "qwen2_vl" \
    --inference_backend "vllm" \
    --enforce_eager true \
    --max_len 4096 \
    --lora_ckpt "./outputs/feedback_gen_grounding_init/"
