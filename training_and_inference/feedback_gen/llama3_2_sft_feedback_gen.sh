export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NPROC_PER_NODE=4 \
# export DS_SKIP_CUDA_CHECK=1
SIZE_FACTOR=8 MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type llama3_2_vision \
    --model Llama-3___2-11B-Vision-Instruct \
    --num_train_epochs 3 \
    --deepspeed zero2 \
    --train_type lora \
    --max_length 4096 \
    --per_device_train_batch_size 4 \
    --truncation_strategy delete \
    --save_strategy epoch \
    --save_only_model True \
    --save_total_limit 3 \
    --output_dir ./outputs/feedback_gen_llama3_2/ \
    --eval_strategy no \
    --dataset ./Dataset/feedback_gen/yellow_images/swift_dataset_train.jsonl \
