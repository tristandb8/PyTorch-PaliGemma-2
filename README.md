# Paligemma 2 (PyTorch)

## Setup


### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/tristandb8/PyTorch-PaliGemma-2.git
   cd PyTorch-PaliGemma-2
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model weights:
   Weights can be found at [HuggingFace](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48)

   For pretrained only model (made for further finetuning)
   ```bash
    git clone https://huggingface.co/google/paligemma2-3b-pt-224
   ```
   For finetuned model, ready to use for general use
   ```bash
    git clone https://huggingface.co/google/paligemma2-3b-ft-docci-448
   ```

   Paligemma 2 can also be found in larger 10b and 28b versions

## Inference

```bash
./launch_inference.sh
```

### Configuration

You can modify the `launch_inference.sh` script to customize the inference parameters:

```bash
#!/bin/bash

MODEL_PATH="../paligemma2-3b-ft-docci-448"  # Path to model weights
PROMPT="what car is this"                   # Text prompt for the model
IMAGE_FILE_PATH="/workspace/images/car.jpg" # Path to the input image
MAX_TOKENS_TO_GENERATE=250                  # Maximum response length
TEMPERATURE=0.8                             # Temperature for sampling
TOP_P=0.9                                   # Top-p sampling parameter
DO_SAMPLE="True"                            # Whether to use sampling
ONLY_CPU="False"                            # Whether to use CPU only

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \
```

## Citation

This work builds upon the following repositories:

```
@misc{gemma_pytorch,
  author = {Google},
  title = {Gemma PyTorch},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/google/gemma_pytorch}
}

@misc{pytorch_paligemma,
  author = {hkproj},
  title = {PyTorch Paligemma},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/hkproj/pytorch-paligemma}
}
```