import os
import re
import torch
import torch.nn as nn

from .config import KnowConfig
from .kar import KAR
from .knowledge import KnowledgeBase, KnowledgeBaseRegistry



class KnowGPT2ForPretraining(KnowGPT2Helper, GPT2ForPreTraining):

    def __init__(self):
        pass
