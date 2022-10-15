import json
import pandas as pd
from utils.my_labelencoder import my_labelencoder

class to_dataframe():
    
    def __init__(self, path):
      self.path = path
      self.df_data = self.to_df()

    def jsonload(self, encoding="utf-8"):
        with open(self.path, encoding=encoding) as f:
            self.j = json.load(f)

    def jsondump(self):
        with open(self.path, "w", encoding="UTF8") as f:
            json.dump(self.j, f, ensure_ascii=False)

    # jsonl 파일 읽어서 list에 저장
    def jsonlload(self, encoding="utf-8"):
        json_list = []
        with open(self.path, encoding=encoding) as f:
            for line in f.readlines():
                json_list.append(json.loads(line))
        self.json_list = json_list
    
    def to_df(self):
      self.jsonlload()
      data = []
      for x in self.json_list:
        data.append([x['sentence_form'], x['annotation'][0][0], x['annotation'][0][2]])
      df_data = pd.DataFrame(data)
      return df_data

    def le (self, mode):
      encoder = my_labelencoder()
      if mode == 'entity':
        encoder.fit('entity')
        self.df_data[1] = encoder.transform(self.df_data[1])

      elif mode == 'sentiment':
        encoder.fit('sentiment')
        self.df_data[2] = encoder.transform(self.df_data[2])
        
      return self.df_data, encoder