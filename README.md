# ComfyUI-GGUF
GGUF Quantization support for native ComfyUI models

This is currently very much WIP. These custom nodes provide support for model files stored in the GGUF format popularized by [llama.cpp](https://github.com/ggerganov/llama.cpp).

While quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs.

![Comfy_Flux1_dev_Q4_0_GGUF_1024](https://github.com/user-attachments/assets/23150750-bcb6-49ef-a78f-9c814528a640)

## Installation

> [!IMPORTANT]  
> Make sure your ComfyUI is on a recent-enough version to support custom ops when loading the UNET-only.

To install the custom node normally, git clone this repository and install the only dependency for inference (`pip install --upgrade gguf`)

```
git clone https://github.com/city96/ComfyUI-GGUF
```

To install the custom node on standalone, open a CMD inside the "ComfyUI_windows_portable" folder (where your `run_nvidia_gpu.bat` file is) and use the following commands:

```
git clone https://github.com/city96/ComfyUI-GGUF ComfyUI/custom_nodes/ComfyUI-GGUF
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
```
## LoRA Supporting

> [!IMPORTANT]  
> Currently only support diffusers_lora format lora (trainning by kohya-ss/sd-script)

![image](https://github.com/user-attachments/assets/d627943f-7fd2-44ec-ad86-96d76804d543)

![image](https://github.com/user-attachments/assets/be206d86-8a5d-4897-bb0a-e1f7fdc8cef5)

test result: https://postimg.cc/62Ngn5X3

tested lora: https://civitai.com/models/638665/shimoe-koharu-flux-dev-lora

## Usage

Simply use the GGUF Unet loader found under the `bootleg` category. Place the .gguf model files in your `ComfyUI/models/unet` folder.

Pre-quantized models:

- [flux1-dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf)
- [flux1-schnell GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf)

> [!WARNING]  
> LoRA / Controlnet / etc are currently not supported due to the weights being quantized.
