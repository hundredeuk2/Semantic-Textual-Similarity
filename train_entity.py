# default
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# For torch
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

# For Transformers
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW

# For Custom
import config
from data_loader.train import custom_dataset, data_loader
from data_loader.predict import pred_data_loader, pred_dataset


class entity_train():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = '/content/drive/MyDrive/Colab Notebooks/corpus'
        self.train = self.path + config.train
        self.test = self.path + config.dev
        self.pred = self.path + config.pred
        self.mode = 'entity'
    
    def set_df(self):
        df, le = data_loader.to_dataframe(self.train).le(self.mode)
        df_dev = data_loader.to_dataframe(self.test).df_data
        df_dev[2] = le.transform(df_dev[2])
        train_dataset, test_dataset = custom_dataset(df, self.mode), custom_dataset(df_dev, self.mode)
        self.le = le
        return train_dataset,test_dataset, le
    

    def model_trains(self):
        train_dataset,test_dataset, le = self.set_df()
        model = AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels = len(le.i2w)).to(self.device)

        epochs = 5
        batch_size = 16
        optimizer = AdamW(model.parameters(), lr=5e-6)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

        losses = []
        accuracies = []

        # wandb.init(project="corpus_sentiment_classifiy", config=config)

        for i in range(epochs):
            total_loss = 0.0
            total_acc = 0
            total = 0
            batches = 0

            model.train()
            p_bar = tqdm(train_loader)
            for input_ids_batch, attention_masks_batch, y_batch in p_bar:
                optimizer.zero_grad()
                # y_batch = F.one_hot(y_batch % 3)
                y_pred = model(input_ids_batch.to(self.device), attention_mask=attention_masks_batch.to(self.device), labels = y_batch.to(self.device))
                loss = y_pred.loss
                loss.backward()
                optimizer.step()

                # total_loss += loss.item()

                _, predicted = torch.max(y_pred.logits, 1)
                predicted = predicted.detach().cpu().numpy()
                correct = accuracy_score(predicted, y_batch.detach().cpu().numpy())
                
                total += len(y_batch)
                total_loss += loss.item()
                total_acc += correct

                batches += 1
                
                p_bar.set_postfix({'loss': loss.item() , 'acc': correct.item()})
                # if batches % 100 == 0:
                #   print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)
            
            losses.append(total_loss)
            accuracies.append(correct)
            print("Train Loss:", total_loss / total, "Accuracy:", total_acc / batches)
        # wandb.log({"Epoch":i,"Accuracy":total_acc / batches, 'Loss': total_loss / total})

        self.testloader = test_loader
        self.model = model
    
    def model_test(self):
        self.model_trains()
        model = self.model
        device = self.device
        test_loader = self.testloader
        model.eval()

        test_correct = 0
        test_total = 0
        predict_data = np.array([])

        for input_ids_batch, attention_masks_batch, y_batch in tqdm(test_loader):
            y_batch = y_batch.to(device)
            y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))
            _, predicted = torch.max(y_pred.logits, 1)
            test_correct += (predicted == y_batch).sum()
            test_total += len(y_batch)
            predict_data = np.hstack([predict_data, predicted.detach().cpu().numpy()])

        print("Accuracy:", test_correct.float() / test_total)

    @torch.no_grad()
    def inference(self, model, test_dataloader):
        device = self.device
        model.eval()
        predict_data = np.array([])

        for input_ids_batch, attention_masks_batch in tqdm(test_dataloader):
            outputs = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))
            _, predicted = torch.max(outputs.logits, 1)
            predict_data = np.hstack([predict_data, predicted.detach().cpu().numpy()])
            
        return predict_data
    
    def make_pred(self):
        pred = self.pred
        mode = self.mode
        
        pred = pred_data_loader(pred)
        pred_dataset = pred_dataset(pred.df_data)
        pred_dataset = DataLoader(pred_dataset, shuffle=False)
    
        predictions = self.inference(self.model, pred_dataset)

        predictions = pd.DataFrame(predictions,columns=[mode])
        predictions = pd.DataFrame(self.le.inverse_transform(predictions[mode]),columns=[mode])
        return predictions
