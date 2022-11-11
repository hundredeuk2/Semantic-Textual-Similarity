import torch
import numpy as np
import pandas as pd
from typing import Callable

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, 
                path: str,
                mode:str,
                tokenizer : Callable,
                max_length : int = 512
    ):
        super().__init__()
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.mode == "train":
            self.text_1, self.text_2, self.target = self.load_data(path)
        else:
            self.text_1, self.text_2 = self.load_data(path)

    def load_data(self, path:str):
        df = pd.read_csv(path)
        text_1 = df['sentence_1'].to_numpy()
        text_2 = df['sentence_2'].to_numpy()
        if self.mode == 'train':
            target = df['label'].to_numpy()
            return text_1, text_2, target
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
        
        if self.mode == "train": #train, val
            return {'input_ids': encoded_dict.input_ids.squeeze(),
                    'attention_mask': encoded_dict.attention_mask.squeeze(), 
                    'labels': torch.tensor(self.target).to(torch.float).unsqueeze(dim=0)}
        
        else: # test
            return {'input_ids': encoded_dict.input_ids.squeeze(),
                    'attention_mask': encoded_dict.attention_mask.squeeze(), 
                    }
        
    def __len__(self):
        return len(self.text_1)