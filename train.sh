MODEL="/mnt2/pretrained_model/LLM/gemma-7b"

CUDA_VISIBLE_DEVICES=0 python3 train_bash.py \
    --stage sft \
    --model_name_or_path $MODEL \
    --do_train True \
    --overwrite_cache True \
    --overwrite_output_dir True \
    --finetuning_type lora \
    --template gemma \
    --dataset_dir data \
    --dataset ai4s \
    --cutoff_len 1536 \
    --learning_rate 5e-05 \
    --num_train_epochs 5.0 \
    --max_samples 2000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 10.0 \
    --logging_steps 50 \
    --save_steps 100 \
    --warmup_steps 0 \
    --flash_attn False \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target q_proj,v_proj \
    --output_dir output \
    --fp16 True \
    --val_size 0.1 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end True \
    --report_to tensorboard \
#     --plot_loss True