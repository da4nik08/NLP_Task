import model
from model import Classifier
import preproces
import numpy as np
import tiktoken
import pandas as pd
import torch
from torch import nn


def get_data(data_path):
    data = pd.read_csv(data_path)
    return data


def save_data(data, data_path):
    data.to_csv(data_path, index=False)  


def make_output_df(reviews, y_pred):
    result = pd.DataFrame()
    result["id"] = reviews["id"]
    sentiment = y_pred.detach().cpu().numpy()
    result["sentiment"] = pd.Series(sentiment).replace([1, 0], ['Positive', 'Negative'])
    return result


def main(input_file, out_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = tiktoken.get_encoding("cl100k_base")
    
    reviews = get_data(input_file)
    
    vfeatures = preproces.preproces(reviews, enc, 150)
    vfeatures = torch.tensor(vfeatures, dtype=torch.long)
    
    model = Classifier(2, 2, 8, 32, 16, 150, enc.n_vocab, dropout_rate=0)
    model.load_state_dict(torch.load("model_name_20230904_222206_299"))
    model.to(device).eval()
    with torch.inference_mode():
        src_padding_mask = (vfeatures == 0)
        y_pred = model(vfeatures.to(device), src_padding_mask.to(device))
    
    soft_out = nn.functional.softmax(y_pred, dim=-1)
    y_pred = soft_out.argmax(dim=-1)
    
    output = make_output_df(reviews, y_pred)
    save_data(output, out_file)