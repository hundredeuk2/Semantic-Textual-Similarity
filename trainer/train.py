import gc
import os
import torch
import wandb
from tqdm.auto import tqdm

class BaseTrainer():
    def __init__(self, model, criterion, metric, optimizer, config, device,
                 train_dataloader, valid_dataloader=None, lr_scheduler=None, epochs=1):
        self.model = model.to(device)
        self.criterion = criterion
        self.metric = metric
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.min_loss_values = 1
    def train(self):
        """
        train_epoch를 돌고 valid_epoch로 평가합니다.
        """
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            self._valid_epoch(epoch)
        torch.cuda.empty_cache()
        del self.model, self.train_dataloader, self.valid_dataloader
        gc.collect()
        
    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        epoch_loss = 0
        steps = 0
        pbar = tqdm(self.train_dataloader)
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            steps += 1

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            label = batch["labels"].to(self.device)
            logits = self.model(input_ids, attention_mask)
            
            loss = self.criterion(logits, label)
            loss.backward()
            epoch_loss += loss.detach().cpu().numpy().item()
            
            self.optimizer.step()
            
            pbar.set_postfix({
                'loss' : epoch_loss / steps
            })
            wandb.log({'train_loss' : epoch_loss / steps})
        print(f"Epoch [{epoch+1}/{self.epochs}] Train_loss : {epoch_loss / steps}")
        pbar.close()

    def _valid_epoch(self, epoch):
        self.model.eval()
        val_loss = 0
        val_steps = 0
        total_val_score = 0
        with torch.no_grad():
            for valid_batch in tqdm(self.valid_dataloader):
                input_ids = valid_batch["input_ids"].to(self.device)
                attention_mask = valid_batch["attention_mask"].to(self.device)
                label = valid_batch["labels"].to(self.device)
                
                logits = self.model(input_ids, attention_mask)

                val_steps += 1
                
                loss = self.criterion(logits, label)
                val_loss += loss.detach().cpu().numpy().item()  
                
                preds = logits.squeeze().detach().cpu().numpy()
                label = label.squeeze().detach().cpu().numpy()
                              
                # total_val_score += self.metric(label, preds).item()
            
            val_loss /= val_steps
            # total_val_score /= val_steps
            print(f"Epoch [{epoch+1}/{self.epochs}] Val_loss : {val_loss}")
            # print(f"Epoch [{epoch+1}/{self.epochs}] Score : {total_val_score}")

            if val_loss <= self.min_loss_values and val_loss <= 0.5:
                print('save checkpoint!')
                if not os.path.exists(f'save/{self.config.model.saved_name}'):
                    os.makedirs(f'save/{self.config.model.saved_name}')
                torch.save(self.model.state_dict(), f'save/{self.config.model.saved_name}/model.pt')
                self.min_loss_values = val_loss

            wandb.log({'val_loss':val_loss})
            # wandb.log({'val_acc':total_val_score})