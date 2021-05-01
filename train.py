import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
from myparser import parse_arguments

from models.UNet import UNet
from dataset import generate_datasets
from trainer import Trainer

from models.metrics import dice_score, iou_score

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_transforms = A.Compose([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50,interpolation=1, border_mode=4, always_apply=False, p=0.5),
            A.HorizontalFlip(always_apply=False, p=0.5),
    ])

    train_dataset, valid_dataset = generate_datasets(config['data_dir'], train_or_test='Train',
                                                     valid_ids=config['val_ids'],
                                                     load_in_memory=config['load_in_memory'], train_transform=train_transforms)

    print(f'Length of training dataset: {len(train_dataset)}')
    print(f'Length of training dataset: {len(valid_dataset)}')

    # TODO: define and add data augmentation + image normalization

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'])
    valid_loader = DataLoader(valid_dataset,
                              config['batch_size'],
                              shuffle=False,
                              num_workers=config['num_workers'])

    model = UNet()
    model = model.to(device)

    if 'load_model' in config:
        model.load_state_dict(torch.load(config['load_model'])['state_dict'])
        print('Loaded model correctly')

    criterion = config['criterion']
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=5-4)
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
    config = parse_arguments()
    main(config)