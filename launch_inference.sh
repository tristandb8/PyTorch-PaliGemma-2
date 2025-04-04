#!/bin/bash

MODEL_PATH="../paligemma2-3b-ft-docci-448"
PROMPT="what car is this"
IMAGE_FILE_PATH="/workspace/images/car.jpg"
MAX_TOKENS_TO_GENERATE=250
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="True"
ONLY_CPU="False"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \

