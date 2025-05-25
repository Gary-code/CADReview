export NPROC_PER_NODE=8 \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model_type  \
    --model  \
    --deepspeed zero3 \
    --num_train_epochs 3 \
    --train_type full \
    --max_length 4096 \
    --per_device_train_batch_size 1 \
    --truncation_strategy delete \
    --save_strategy epoch \
    --save_only_model True \
    --output_dir ./outputs/alignment_caption/ \
    --eval_strategy no \
    --dataset ./Dataset/caption/swift_dataset_train_extend.jsonl \
