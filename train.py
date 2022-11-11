import torch
import argparse
from seed import seed_everything
from wandb_setting import wandb_setting

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data_loader.loader import CustomDataset
import model.model as Model
import model.metric as Metric
import model.loss as Criterion
import torch.optim as optim
import trainer.train as Trainer

def main(config):
    seed_everything(config.train.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    
    # 데이터셋 로드 클래스를 불러옵니다.
    train = CustomDataset(mode = "train", path = config.data.train_path, tokenizer=tokenizer, max_length=config.train.max_length)  
    valid = CustomDataset(mode = "train", path = config.data.valid_path, tokenizer=tokenizer, max_length=config.train.max_length)
    
    train_dataloader = DataLoader(train, batch_size= config.train.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid, batch_size= config.train.batch_size, shuffle=False)
    
    # GPU 사용 설정을 해줍니다.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 아키텍처를 불러옵니다.
    print(f'현재 적용되고 있는 모델은 {config.model.model_class}입니다.')
    model = getattr(Model, config.model.model_class)\
            (model_name = config.model.model_name, drop_rate = config.train.dropout_rate, num_labels = config.model.num_classes)\
            .to(device)

    criterion = getattr(Criterion, config.model.loss)
    metric = getattr(Metric, config.model.metric)
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
    
    lr_scheduler = None
    epochs = config.train.max_epoch
    
    trainer = getattr(Trainer, 'BaseTrainer')(
            model = model,
            criterion = criterion,
            metric = metric,
            optimizer = optimizer,
            config = config,
            device = device,
            train_dataloader = train_dataloader,
            valid_dataloader = valid_dataloader,
            lr_scheduler=lr_scheduler,
            epochs=epochs
        )
    
    trainer.train()

if __name__=='__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()

    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')
    config_w = wandb_setting(entity = "naver-inapp",
                             project = "test-py",
                             group_name = "electra",
                             experiment_name = "Baseline_model",
                             arg_config = args.config)
    main(config_w)
    