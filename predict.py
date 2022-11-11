from unittest.util import _MAX_LENGTH
import torch
import streamlit as st
from model.model import EncoderModel as Model
import yaml
from typing import Tuple
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from predict_loader import CustomDataset

@st.cache
def load_model():
    with open("configs/config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model(config['model']['model_name'],num_labels=1, drop_rate=0.2).to(device)
    model.load_state_dict(torch.load('save/model.pt', map_location=device))

    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['model_name'],
        use_fast=True
    )

    return model, tokenizer


def get_prediction(data_1, data_2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model()
    data = pd.DataFrame({'sentence_1':[data_1],
                        'sentence_2' : [data_2]})
    test_dataset = CustomDataset(data, tokenizer)
    test_dataloader = DataLoader(test_dataset,
            batch_size=1,
            shuffle=False)

    with torch.no_grad():
        for batch in test_dataloader:
            output = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            pred = output.squeeze().detach().cpu().numpy().item()
            pred = round(pred, 1)
    return pred