import json
import pandas as pd

class to_dataframe_pred():

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
            data.append(x['sentence_form'])
        df_data = pd.DataFrame(data)
        return df_data