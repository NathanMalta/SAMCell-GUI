import cv2
import math
import monai
import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import numpy as np
import wandb

from transformers import SamModel, SamConfig, SamMaskDecoderConfig
from transformers.models.sam.modeling_sam import SamMaskDecoder, SamVisionConfig
from transformers.models.sam import convert_sam_original_to_hf_format

class FinetunedSAM():
    '''a helper class to handle setting up SAM from the transformers library for finetuning
    '''
    def __init__(self, sam_model, finetune_vision=False, finetune_prompt=True, finetune_decoder=True):
        self.model = SamModel.from_pretrained(sam_model)
        #freeze required layers
        if not finetune_vision:
            for param in self.model.vision_encoder.parameters():
                param.requires_grad_(False)
        else:
            for param in self.model.vision_encoder.parameters():
                param.requires_grad_(True)
            
        if not finetune_prompt:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad_(False)
        
        if not finetune_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad_(False)
        else:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad_(True)

    def get_model(self):
        return self.model
    
    def load_weights(self, weight_path, map_location=None):
        self.model.load_state_dict(torch.load(weight_path, map_location=map_location))