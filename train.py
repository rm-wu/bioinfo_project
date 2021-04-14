import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import argparse

from model.UNet import UNet
from dataset import generate_datasets
from trainer import Trainer

from model.metrics import dice_score, iou_score

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset, valid_dataset = generate_datasets(config['data_dir'],
                                                     valid_ids=config['val_ids'],
                                                     load_in_memory=config['load_in_memory'])
    # TODO: define and add data augmentation + image normalization
    # train_dataset.transform = train_transform
    # valid_dataset.transform = valid_transform
    transforms = A.Compose(
        [
            A.Normalize(),  # TODO: change values
            ToTensorV2()
        ]
    )
    train_dataset.transform = transforms
    valid_dataset.transform = transforms

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset,
                              config['batch_size'],
                              shuffle=False,
                              num_workers=config['num_workers']
                              )
    model = UNet()
    model = model.to(device)

    criterion = config['criterion']
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=3e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    metrics = [iou_score, dice_score]

    trainer = Trainer(model=model,
                      criterion=criterion,
                      metrics=metrics,
                      optimizer=optimizer,
                      lr_scheduler=scheduler,
                      config=config,
                      train_loader=train_loader,
                      val_loader=valid_loader,
                      device=device)
    trainer.train()

    return model


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Biomedical Image Segmentation with UNet and HQA")
    '''
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default:None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to the latest checkpoint (default:None)')
    '''
    args.add_argument('-d', '--data', type=str,
                      help='Data path')
    args.add_argument('--colab', action='store_true')
    args = args.parse_args()
    colab = args.colab
    # TODO: args --> config or json file

    config = dict()
    if colab:
        config['data_dir'] = '/content/drive/My Drive/Bioinformatics/dataset'
        config['num_workers'] = 4
        config['load_in_memory'] = True
    else:
        config['data_dir'] = args.data
        config['num_workers'] = 1
        config['load_in_memory'] = False

    # TODO: check if this loss is good
    config['criterion'] = nn.BCEWithLogitsLoss()

    # TODO: add other metrics like accuracy etc.
    # TODO: configure the optimizer/LR Scheduler and their hyperparams
    config['val_ids'] = ['1', '5']
    config['epochs'] = 20
    config['batch_size'] = 4

    main(config)
