export HF_ENDPOINT=https://hf-mirror.com

python -u train.py \
    --data-root ../data/processed_dataset \
    --output-dir ./pix2pix-lora-run \
    --learning-rate 3e-5 \
    --max-train-steps 2000 \
    --lora-rank 8 \
    --image-size 96 \
    > train.log 2>&1 &