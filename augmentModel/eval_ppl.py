import torch
import torch.nn as nn
from src.model.model import KnowGPT2Model, KnowGPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.knowledge.khangraph import KhanGraph
from src.datasets.khan_academy import KhanAcademyMathDataset
from src.datasets.mathematica_with_steps import MathematicaWithStepsMathDataset
import os
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from datetime import datetime



def load_data(loc):

    ts = []
    with os.scandir(loc) as scripts:
        for script in scripts:
            with open(script, 'r') as input:
                ts.append(input.read())

    text = "\n\n".join(ts)

    return text



def get_tokenizer_gpt():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return tokenizer


def eval_ppl():

    device = "cuda"
    model_id = "/home/besperk/Code/math/checkpoints/TEMP/04-23-2022__13:40:48/gpt2_amps"

    # Establish Model & Tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    # Load Dataset
    test = load_data("./Vault/waste/")
    tokenizer = get_tokenizer_gpt()
    encodings = tokenizer(test, return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print(ppl)



# Test
if __name__=="__main__":
    eval_ppl()
