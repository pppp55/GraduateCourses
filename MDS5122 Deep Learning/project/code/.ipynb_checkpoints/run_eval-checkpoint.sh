export HF_ENDPOINT=https://hf-mirror.com

# python -u eval.py --data-root ../data/processed_dataset \
#     --lora-weights ./pix2pix-lora-run/lora-step-1500/ \
#     --image-size 96 \
#     --no-use-lora \
#     --save-images \
#     > eval.log 2>&1 &

python -u eval.py --data-root ../data/processed_dataset \
    --lora-weights ./pix2pix-lora-run/lora-step-1500/ \
    --image-size 96 \
    --save-images \
    > eval.log 2>&1 &