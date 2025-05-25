bash ./my_experiment/vllm_deploy.sh \
    --cuda_devices "0,1,2,3,4,5,6,7" \
    --model_path "./outputs/alignment_grounding/llava_ov/" \
    --template "llava_onevision_hf" \
    --inference_backend "vllm" \
    --enforce_eager true \
    --max_len 12000 \
    --lora_ckpt "./outputs/feedback_gen_grounding_init/llava_ov/"
