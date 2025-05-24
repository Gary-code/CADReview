#!/bin/bash

# 默认参数
DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
DEFAULT_MODEL_PATH="qwen2-vl-7b-instruct"
DEFAULT_TEMPLATE="qwen2-vl-7b-instruct"
DEFAULT_INFERENCE_BACKEND="vllm"
DEFAULT_VLLM_ENFORCE_EAGER=true
DEFAULT_VLLM_MAXLEN=2280
DEFAULT_LORA_CKPT=""
DEFAULT_MAX_SEQUENCE_LEN=96
DEFAULT_TENSOR_PARALLEL_SIZE=1
# DEFAULT_MAX_IMAGE_NUMS='{"image":4,"video":2}'
DEFAULT_MAX_IMAGE_NUMS='{"image":2}'
# 参数解析函数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cuda_devices)
                CUDA_VISIBLE_DEVICES="$2"
                shift 2
                ;;
            --model_path)
                MODEL_PATH="$2"
                shift 2
                ;;
            --template)
                TEMPLATE="$2"
                shift 2
                ;;
            --inference_backend)
                INFERENCE_BACKEND="$2"
                shift 2
                ;;
            --enforce_eager)
                VLLM_ENFORCE_EAGER="$2"
                shift 2
                ;;
            --max_len)
                VLLM_MAXLEN="$2"
                shift 2
                ;;
            --lora_ckpt)
                LORA_CKPT="$2"
                shift 2
                ;;
            --max_num_seqs)
                MAX_SEQUENCE_LEN="$2"
                shift 2
                ;;
            --tensor_parallel_size)
                TENSOR_PARALLEL_SIZE="$2"
                shift 2
                ;;
            --limit_mm_per_prompt)
                MAX_IMAGE_NUMS="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                exit 1
                ;;
        esac
    done
}

# 使用默认值初始化参数
CUDA_VISIBLE_DEVICES=$DEFAULT_CUDA_VISIBLE_DEVICES
MODEL_PATH=$DEFAULT_MODEL_PATH
TEMPLATE=$DEFAULT_TEMPLATE
INFERENCE_BACKEND=$DEFAULT_INFERENCE_BACKEND
VLLM_ENFORCE_EAGER=$DEFAULT_VLLM_ENFORCE_EAGER
VLLM_MAXLEN=$DEFAULT_VLLM_MAXLEN
LORA_CKPT=$DEFAULT_LORA_CKPT
MAX_SEQUENCE_LEN=$DEFAULT_MAX_SEQUENCE_LEN
TENSOR_PARALLEL_SIZE=$DEFAULT_TENSOR_PARALLEL_SIZE
MAX_IMAGE_NUMS=$DEFAULT_MAX_IMAGE_NUMS
# 调用参数解析
parse_args "$@"

# 打印配置供检查
echo "Using the following configurations:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH: $MODEL_PATH"
echo "TEMPLATE: $TEMPLATE"
echo "INFERENCE_BACKEND: $INFERENCE_BACKEND"
echo "VLLM_ENFORCE_EAGER: $VLLM_ENFORCE_EAGER"
echo "VLLM_MAXLEN: $VLLM_MAXLEN"
echo "LORA_CKPT: $LORA_CKPT"
echo "MAX_SEQUENCE_LEN: $MAX_SEQUENCE_LEN"
echo "TENSOR_PARALLEL_SIZE: $TENSOR_PARALLEL_SIZE"
echo "MAX_IMAGE_NUMS: $MAX_IMAGE_NUMS"

# 循环，根据 tensor_parallel_size 分组
IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"

num_gpus=${#GPU_ARRAY[@]}
group_size=$TENSOR_PARALLEL_SIZE

if (( num_gpus % group_size != 0 )); then
    echo "Error: Number of GPUs ($num_gpus) is not divisible by TENSOR_PARALLEL_SIZE ($group_size)."
    exit 1
fi

group_count=$((num_gpus / group_size))

for group in $(seq 0 $((group_count - 1))); do
    start_idx=$((group * group_size))
    end_idx=$((start_idx + group_size - 1))

    gpu_subset=("${GPU_ARRAY[@]:start_idx:group_size}")
    gpu_subset_str=$(IFS=','; echo "${gpu_subset[*]}")

    export CUDA_VISIBLE_DEVICES=$gpu_subset_str
    port=$((9010 + group))

    echo "Launching on GPUs: $gpu_subset_str with port $port"

    if [ -z "$LORA_CKPT" ]; then
        # 未设置 LORA_CKPT
        swift deploy \
            --model_type $TEMPLATE \
            --merge_lora false \
            --port $port \
            --enforce-eager $VLLM_ENFORCE_EAGER \
            --max_model_len $VLLM_MAXLEN \
            --model $MODEL_PATH \
            --infer_backend $INFERENCE_BACKEND \
            --max_num_seqs $MAX_SEQUENCE_LEN \
            --limit_mm_per_prompt $MAX_IMAGE_NUMS \
            --tensor_parallel_size $TENSOR_PARALLEL_SIZE &
    else
        # 设置了 LORA_CKPT
        swift deploy \
            --model_type $TEMPLATE \
            --merge_lora true \
            --ckpt_dir $LORA_CKPT \
            --port $port \
            --enforce-eager $VLLM_ENFORCE_EAGER \
            --max_model_len $VLLM_MAXLEN \
            --model $MODEL_PATH \
            --infer_backend $INFERENCE_BACKEND \
            --max_num_seqs $MAX_SEQUENCE_LEN \
            --limit_mm_per_prompt $MAX_IMAGE_NUMS \
            --tensor_parallel_size $TENSOR_PARALLEL_SIZE &
    fi
done

wait
