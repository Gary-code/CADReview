export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NPROC_PER_NODE=8 \
# export DS_SKIP_CUDA_CHECK=1
SIZE_FACTOR=8 MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model_type qwen2_vl \
    --model ./outputs/alignment_code_completion/qwen2_vl_grounding_init/\
    --deepspeed zero3 \
    --num_train_epochs 3 \
    --train_type lora \
    --max_length 6144 \
    --per_device_train_batch_size 1\
    --truncation_strategy delete \
    --save_strategy epoch \
    --save_only_model True \
    --output_dir ./outputs/code_editor_code_completion_init/ \
    --eval_strategy no \
    --dataset ./Dataset/code_editor/two_images/swift_dataset_train.jsonl \
