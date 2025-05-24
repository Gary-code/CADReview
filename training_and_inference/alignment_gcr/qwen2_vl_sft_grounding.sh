export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NPROC_PER_NODE=8 \
# export DS_SKIP_CUDA_CHECK=1
SIZE_FACTOR=8 MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model_type qwen2_vl \
    --model qwen2-vl-7b-instruct \
    --deepspeed zero3 \
    --num_train_epochs 1 \
    --train_type lora \
    --target_modules qkv q_proj k_proj v_proj \
    --max_length 4096 \
    --per_device_train_batch_size 1 \
    --truncation_strategy delete \
    --save_strategy steps \
    --save_steps 4000 \
    --save_only_model True \
    --output_dir ./outputs/alignment_grounding/ \
    --eval_strategy no \
    --dataset ./Dataset/grounding/swift_dataset_train_extend.jsonl \
