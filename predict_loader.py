import torch
import numpy as np
import pandas as pd
from typing import Callable

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, 
                df: str,
                tokenizer : Callable,
                max_length : int = 128
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.text_1, self.text_2 = self.load_data(df)

    def load_data(self, df):
        text_1 = df['sentence_1'].to_numpy()
        text_2 = df['sentence_2'].to_numpy()
        return text_1, text_2

    def __getitem__(self, idx):
        sentence_1 = self.text_1
        sentence_2 = self.text_2
        
        encoded_dict = self.tokenizer.encode_plus(
            sentence_1[idx],
            sentence_2[idx],           
            add_special_tokens = True,      
            max_length = self.max_length,           
            pad_to_max_length = True,
            truncation=True,
            return_attention_mask = True,   
            return_tensors = 'pt',          
            )
    
        
        return {'input_ids': encoded_dict.input_ids.squeeze(),
                    'attention_mask': encoded_dict.attention_mask.squeeze(), 
                    }
        
    def __len__(self):
        return len(self.text_1)