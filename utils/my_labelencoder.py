class my_labelencoder():
    def __init__(self):
        self.entity_dict = {'제품 전체#품질', '패키지/구성품#디자인', '본품#일반', '제품 전체#편의성', '본품#다양성', 
                            '제품 전체#디자인', '패키지/구성품#가격', '본품#품질', '브랜드#인지도', '제품 전체#일반',
                            '브랜드#일반', '패키지/구성품#다양성', '패키지/구성품#일반', '본품#인지도', '제품 전체#가격',
                            '본품#편의성', '패키지/구성품#편의성', '본품#디자인', '브랜드#디자인', '본품#가격', '브랜드#품질', 
                            '제품 전체#인지도', '패키지/구성품#품질', '제품 전체#다양성', '브랜드#가격'}
        self.sentiment_dict = {'negative', 'neutral','positive'}

    def fit(self, mode):
        i2w = dict()
        w2i = dict()
        if mode == "entity":
            for idx, dic in enumerate(self.entity_dict):
                i2w[idx] = dic
                w2i[dic] = idx

        elif mode == "sentiment":
            for idx, dic in enumerate(self.sentiment_dict):
                i2w[idx] = dic
                w2i[dic] = idx
                
        self.i2w, self.w2i = i2w, w2i
    
    def transform(self, data):
        data = data.values
        tmp = []
        for x in data:
            tmp.append(self.w2i[x])
        return tmp
    
    def inverse_transform(self, data):
        data = data
        tmp = []
        for x in data:
            tmp.append(self.i2w[x])
        return tmp