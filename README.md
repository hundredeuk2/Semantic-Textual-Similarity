# Semantic-Textual-Similarity

## Introduce
![STS_demo](https://user-images.githubusercontent.com/73874591/201269490-158b73f7-85c5-4c41-a39f-0a6099ce4c4d.gif)

## Wrap up Report
Please Click [here](https://www.notion.so/NLP-Wrap-up-Report-49002b7186304491a7954e0d4e7ae309) for more details!

## [NLP] Competition Summary

<img src="https://user-images.githubusercontent.com/97590480/205296373-f04d2ba4-6232-457d-8b2e-4bcbafa1a353.png">

Thie competition is two sentence classification for STS(Semantic Text Similairy). STS deals with scoring for how similar the two sentences are semantically.

___

## Members
|김근형|김찬|권용훈|유선종|이헌득|
|:---:|:---:|:---:|:---:|:---:|
|<img src="https://user-images.githubusercontent.com/97590480/205299519-174ef1be-eed6-4752-9f3d-49b64de78bec.png">|<img src="https://user-images.githubusercontent.com/97590480/205299316-ea3dc16c-00ec-4c37-b801-3a75ae6f4ca2.png">|<img src="https://user-images.githubusercontent.com/97590480/205299125-c4e55849-6555-4c9b-908a-0341e2b6fa22.png">|<img src="https://user-images.githubusercontent.com/97590480/205299037-aec039ea-f8d3-46c6-8c11-08c4c88e4c56.jpeg">|<img src="https://user-images.githubusercontent.com/97590480/205299457-5292caeb-22eb-49d2-a52e-6e69da593d6f.jpeg">|
|[Github](https://github.com/kimkeunhyeong)|[Github](https://github.com/chanmuzi)|[Github](https://github.com/kwon13)|[Github](https://github.com/Trailblazer-Yoo)|[Github](https://github.com/hundredeuk2)|

## Project Tree

<img src="https://user-images.githubusercontent.com/97590480/205296028-741f9042-187a-40e8-a774-a024864c0b9c.png">

___

## Usage

### Demo For STS Input
```
streamlit run app.py
```
### train
add command `config.yaml` **file name** in configs directory. We provide three config files(koelectra_GRU(3), T5_encoder_GRU(3), T5).
```
python3 train.py --config koelectra_GRU3
```

### Inference
```
python3 test.py --config koelectra_GRU3
```

___

### Result

<img src="https://user-images.githubusercontent.com/97590480/205300430-04ddc8a0-c0b5-4a3a-af31-1088d7af40c9.png">