import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import numpy as np

from transformers import SamModel, SamConfig, SamMaskDecoderConfig
from transformers.models.sam.modeling_sam import SamMaskDecoder, SamVisionConfig
from transformers.models.sam import convert_sam_original_to_hf_format

class FinetunedSAM():
    '''a helper class to handle setting up SAM from the transformers library for finetuning
    '''
    def __init__(self, sam_model):
        self.model = SamModel.from_pretrained(sam_model)
        self.model.eval()

    def get_model(self):
        return self.model
    
    def load_weights(self, weight_path, map_location=None):
        self.model.load_state_dict(torch.load(weight_path, map_location=map_location))