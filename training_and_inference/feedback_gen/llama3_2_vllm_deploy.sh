bash ./my_experiment/vllm_deploy.sh \
    --cuda_devices "0,1,2,3" \
    --model_path "Llama-3___2-11B-Vision-Instruct" \
    --template "llama3_2_vision" \
    --inference_backend "vllm" \
    --enforce_eager true \
    --max_len 4096 \
    --lora_ckpt "./outputs/feedback_gen_llama3_2/"
