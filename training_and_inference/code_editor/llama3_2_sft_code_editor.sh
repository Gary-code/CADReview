# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
export NPROC_PER_NODE=8 \
# export DS_SKIP_CUDA_CHECK=1
SIZE_FACTOR=8 MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type llama3_2_vision \
    --model Llama-3___2-11B-Vision-Instruct \
    --deepspeed zero2 \
    --num_train_epochs 3 \
    --train_type lora \
    --max_length 6144 \
    --per_device_train_batch_size 4 \
    --truncation_strategy delete \
    --save_strategy epoch \
    --save_only_model True \
    --output_dir ./outputs/code_editor_directly_llama/ \
    --eval_strategy no \
    --dataset ./Dataset/code_editor_directly/swift_dataset_train.jsonl \
