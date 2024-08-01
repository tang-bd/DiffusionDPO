export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="/data/bingda/coco/augmented_coco_karpathy_train.csv"
export TRAIN_DATA_DIR="/data/bingda/coco"


# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --train_batch_size=4 \
  --dataloader_num_workers=32 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=200 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=50 \
  --learning_rate=1e-8 --scale_lr \
  --checkpointing_steps 20 \
  --beta_dpo 5000 \
  --sdxl  \
  --resume_from_checkpoint /data/bingda/dpo/checkpoint-100 \
  --output_dir="/data/bingda/dpo" \
  
