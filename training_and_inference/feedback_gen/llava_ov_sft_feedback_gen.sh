export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NPROC_PER_NODE=8 \
# export DS_SKIP_CUDA_CHECK=1
SIZE_FACTOR=8 MAX_PIXELS=602112 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model_type llava_onevision_hf \
    --model ./outputs/alignment_grounding/llava_ov/ \
    --deepspeed zero3 \
    --num_train_epochs 3 \
    --train_type lora \
    --max_length 4096 \
    --per_device_train_batch_size 1 \
    --truncation_strategy delete \
    --save_strategy epoch \
    --save_only_model True \
    --output_dir ./outputs/feedback_gen_grounding_init/llava_ov \
    --eval_strategy no \
    --dataset ./Dataset/feedback_gen/yellow_images/swift_dataset_train.jsonl \
