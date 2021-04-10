import torch
import numpy as np
import argparse
import model.UNet as UNet

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Biomedical Image Segmentation with UNet and HQA")
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default:None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to the latest checkpoint (default:None)')
    args = args.parse_args()
    # TODO: args --> config
    config = None
    main(config)