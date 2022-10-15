from torch.utils.data import Dataset
from transformer import AutoTokenizer

class CustomDataset(Dataset):
    
    def __init__(self, dataframe, mode):
      self.dataset = dataframe
      self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
      self.mode = mode

      print(self.dataset.describe())
  
    def __len__(self):
      return len(self.dataset)
    
    def __getitem__(self, idx):
      row = self.dataset.iloc[idx,:].values
      text = row[0]
      if self.mode == "sentiment":
        y = row[2]
      elif self.mode == 'entity':
        y = row[1]

      inputs = self.tokenizer(
          text, 
          return_tensors='pt',
          truncation=True,
          max_length=150,
          padding='max_length',
          # pad_to_max_length=True,
          add_special_tokens=True,
          return_attention_mask=True
          )
      
      input_ids = inputs['input_ids'][0]
      attention_mask = inputs['attention_mask'][0]

      return input_ids, attention_mask, y
    #################################