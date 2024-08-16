# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import gguf
import logging

import comfy.sd
import comfy.utils
import comfy.model_management
import folder_paths

import safetensors
import uuid

from .ops import GGMLTensor, GGMLOps

# TODO: This causes gguf files to show up in the main unet loader
folder_paths.folder_names_and_paths["unet"][1].add(".gguf")

def gguf_sd_loader(path):
    """
    Read state dict as fake tensors
    """
    reader = gguf.GGUFReader(path)
    sd = {}
    dt = {}
    for tensor in reader.tensors:
        sd[str(tensor.name)] = GGMLTensor(tensor)
        dt[str(tensor.tensor_type)] = dt.get(str(tensor.tensor_type), 0) + 1

    # sanity check debug print
    print("\nggml_sd_loader:")
    for k,v in dt.items():
        print(f" {k:30}{v:3}")
    print("\n")
    return sd

class UnetLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet") if x.endswith(".gguf")]
        return {
            "required": {
                "unet_name": (unet_names,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "bootleg"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, unet_name):
        unet_path = folder_paths.get_full_path("unet", unet_name)
        sd = gguf_sd_loader(unet_path)
        ggmlops = GGMLOps()
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ggmlops}
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        model.ggmlops = ggmlops
        return (model, )

class UnetGGUFLora:
    def __init__(self) -> None:
        self.id = str(uuid.uuid4())
        self.ggmlops = None
        # print('init:',self.id  )
    def __del__(self):
        # print('remove:',self.id  )
        if self.ggmlops is not None:
            del self.ggmlops.Linear.loras[self.id]
            del self.ggmlops.Linear.loras_strength[self.id]
            
        
    @classmethod
    def INPUT_TYPES(s):
        
        lora_names = [x for x in folder_paths.get_filename_list("loras") if x.endswith(".safetensors")]
        lora_names.insert(0, "None")
        return {
            "required": {
                "unet_model": ("MODEL",),
                "lora_name": (lora_names,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "bootleg"
    TITLE = "Unet Lora Loader (GGUF)"
    
    def load_lora(self, unet_model, lora_name, strength_model):
        print(lora_name)
        if lora_name != 'None':
            lora_tensor = {}
            with safetensors.safe_open(folder_paths.get_full_path("loras", lora_name), framework="pt", device='cuda') as f:
                for k in f.keys():
                    lora_tensor[k] = f.get_tensor(k)
            self.ggmlops = unet_model.ggmlops
            self.ggmlops.Linear.loras[self.id] = lora_tensor
            self.ggmlops.Linear.loras_strength[self.id] = strength_model
            
        return (unet_model, lora_name, strength_model, )            
                    

NODE_CLASS_MAPPINGS = {
    "UnetLoaderGGUF": UnetLoaderGGUF,
    "UnetGGUFLora": UnetGGUFLora,
}
