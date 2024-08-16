# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import gguf
import torch
import numpy as np

import comfy.ops
from .dequant import dequantize_tensor

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, tensor, *args, **kwargs):
        super().__init__()
        self.tensor_type = tensor.tensor_type
        self.tensor_shape = torch.Size(
            np.flip(list(tensor.shape))
        )

    def __new__(cls, tensor, *args, **kwargs):
        data = torch.tensor(tensor.data)
        return super().__new__(cls, data, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = self.tensor_type
        new.tensor_shape = self.tensor_shape
        return new

    @property
    def shape(self):
        return self.tensor_shape

class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = None
        self.bias = None
        self.key_name = None
        self.lora_WA_k_name = None
        self.lora_WB_k_name = None
        self.lora_alpha_k_name = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for k,v in state_dict.items():
            # print("KEY:",k)
            
            if k.startswith('single_blocks.') and k[len(prefix):] == "weight":
                key_name_in_base_model = k[len('single_blocks.'):-len('.weight')].replace(".", "_")             # e.g: single_blocks.30.linear2.weight
                self.lora_WA_k_name = 'lora_unet_single_blocks_' + key_name_in_base_model + '.lora_up.weight'   #      lora_unet_single_blocks_30_linear2.lora_up.weight
                self.lora_WB_k_name = 'lora_unet_single_blocks_' + key_name_in_base_model + '.lora_down.weight' #      lora_unet_single_blocks_30_linear2.lora_down.weight
                self.lora_alpha_k_name = 'lora_unet_single_blocks_' + key_name_in_base_model  + '.alpha'

            elif k.startswith('double_blocks.') and k[len(prefix):] == "weight":
                key_name_in_base_model = k[len('double_blocks.'):-len('.weight')].replace(".", "_")
                self.lora_WA_k_name = 'lora_unet_double_blocks_' + key_name_in_base_model + '.lora_up.weight'
                self.lora_WB_k_name = 'lora_unet_double_blocks_' + key_name_in_base_model + '.lora_down.weight'
                self.lora_alpha_k_name = 'lora_unet_double_blocks_' + key_name_in_base_model + '.alpha'
                
            self.key_name = k
            
            if k[len(prefix):] == "weight":
                self.weight = v
            elif k[len(prefix):] == "bias":
                self.bias = v
            else:
                missing_keys.append(k)

    def _apply(self, fn):
        if self.weight is not None:
            self.weight = fn(self.weight)
        if self.bias is not None:
            self.bias = fn(self.bias)
        super()._apply(fn)
        return self

    def get_weights(self, dtype=torch.float16):
        weight = dequantize_tensor(self.weight, dtype)
        bias = dequantize_tensor(self.bias, dtype)
        return (weight, bias)

class GGMLOps(comfy.ops.disable_weight_init):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.Linear.loras = {}
        self.Linear.loras_strength = {}
        
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer):
        def __init__(self, *args, device=None, dtype=None, **kwargs):
            super().__init__(device=device, dtype=dtype)
            self.parameters_manual_cast = torch.float32
        def forward(self, x):
            weight, bias = self.get_weights(x.dtype)
            
            
            for lora_blk, v in self.loras.items():
                if self.lora_WA_k_name in v:
                    strength = self.loras_strength[lora_blk]
                    lora_WA_k = v[self.lora_WA_k_name]
                    lora_WB_k = v[self.lora_WB_k_name]
                    if self.lora_alpha_k_name in v:
                        alpha = v[self.lora_alpha_k_name].item()
                        alpha /= lora_WB_k.shape[0]
                    else:
                        alpha = 1.0
                    
                    lora_WA_k = lora_WA_k.to(weight.device).to(x.dtype)
                    lora_WB_k = lora_WB_k.to(weight.device).to(x.dtype)
                    lora_diff = (strength * alpha) * torch.matmul(lora_WA_k, lora_WB_k)
                    weight += lora_diff
                    del lora_diff, lora_WA_k, lora_WB_k
            x = torch.nn.functional.linear(x, weight, bias)
            del weight, bias
            return x
