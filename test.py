import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import MetricMonitor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from myparser import parse_arguments
from model.UNet import UNet
from dataset import generate_datasets
from model.metrics import dice_score, iou_score
from tqdm import tqdm
from torchvision.transforms import CenterCrop


# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

TILE_DIM = 400

def main(config):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(config['data_dir'])
    test_dataset = generate_datasets(config['data_dir'], train_or_test='Test',
                                                     valid_ids=config['val_ids'],
                                                     load_in_memory=config['load_in_memory'])

    print(f'Length of test dataset: {len(test_dataset)}')

    # TODO: define and add data augmentation + image normalization
    # train_dataset.transform = train_transform
    # valid_dataset.transform = valid_transform
    transforms = A.Compose(
        [
            A.Normalize(),  # TODO: change values
            ToTensorV2()
        ]
    )
    test_dataset.transform = transforms

    test_loader = DataLoader(test_dataset,
                              batch_size=config['batch_size'],
                              shuffle=False,
                              num_workers=config['num_workers'])

    model = UNet()
    model = model.to(device)
    model.eval()

    if 'load_model' in config:
        model.load_state_dict(torch.load(config['load_model'])['state_dict'])
        print('Loaded model correctly')
    else:
        print('No model loaded!')
        return

    file_tested_model_name=config['load_model'].split('/')[-2]+'_'+config['load_model'].split('/')[-1]
    tested_model_folder=config['save_test_dir']

    f=open(tested_model_folder+file_tested_model_name.replace('.pth','')+'.txt', 'w')

    metrics = [iou_score, dice_score]
    center=CenterCrop(TILE_DIM)
    criterion=config['criterion']

    stream = tqdm(test_loader)
    metric_monitor = MetricMonitor()
    with torch.no_grad():
        for batch_idx, (image, mask) in enumerate(stream):

            image, mask = image.to(device), mask.to(device)
            output = model(image)
            loss = criterion(center(output), center(mask))
            metric_monitor.update("Loss", loss.item())
            for metric in metrics:
                metric_monitor.update(metric.__name__, metric(center(output),
                                                              center(mask)))
            stream.set_description(f"Testing\t|{metric_monitor}")

    f.write(f'Loss_test: {metric_monitor.return_value("Loss")} \n')

    for metric in metrics:
        f.write(f"{metric.__name__}_Test: {metric_monitor.return_value(metric.__name__)} \n")

    f.close()

if __name__ == '__main__':
    config = parse_arguments()
    main(config)