import argparse
import torch.nn as nn
from models.loss import iou_loss

def parse_arguments():
    args = argparse.ArgumentParser(
        description="Biomedical Image Segmentation with UNet and HQA"
    )
    '''
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default:None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to the latest checkpoint (default:None)')
    '''
    args.add_argument('-d', '--data', type=str,
                      help='Data path')
    args.add_argument("--loss", type=str, default="iou",
                      choices=["bce", "iou"], help="Loss for the model")
    args.add_argument('--colab', action='store_true')
    args.add_argument('--windows', action='store_true')

    args = args.parse_args()
    colab = args.colab
    windows = args.windows

    # TODO: args --> config or json file

    config = dict()

    #################################################
    #           Colab OPTIONS                       #
    #   Change if necessary                         #
    #################################################
    if colab:
        config['data_dir'] = '/content/drive/My Drive/Bioinformatics/dataset'
        config['num_workers'] = 4
        config['load_in_memory'] = True
        config['save_dir'] = '/content/drive/My Drive/Bioinformatics/bioinfo_project/runs'
        config['val_ids'] = ['1', '5']

    #################################################
    #           Windows OPTIONS                     #
    #   Change if necessary                         #
    #################################################
    elif windows:
        config['data_dir'] = 'C:/Users/emanu/Documents/Polito/Bioinformatics/dataset/'
        config['num_workers'] = 1
        config['load_in_memory'] = False
        config['val_ids'] = ['1']
        config['save_dir'] = 'C:/Users/emanu/Documents/Polito/Bioinformatics/dir'
        config['save_test_dir']='C:/Users/emanu/Documents/Polito/Bioinformatics/test/'
        # config['tensorboard_dir']='C:/Users/emanu/Documents/Polito/Bioinformatics/tensorboard'
    #################################################
    #           Local OPTIONS                       #
    #   Change if necessary                         #
    #################################################
    else:
        config['data_dir'] = args.data
        config['num_workers'] = 1
        config['load_in_memory'] = False
        config['val_ids'] = ['1', '5']
        config['save_dir'] = './runs'

    # TODO: check if this loss is
    if args.loss == "bce":
        config['criterion'] = nn.BCEWithLogitsLoss()
    elif args.loss == "iou":
        config['criterion'] = iou_loss

    # TODO: add other metrics like accuracy etc.
    # TODO: configure the optimizer/LR Scheduler and their hyperparams
    config['epochs'] = 2
    config['batch_size'] = 4
    config['trainer'] = {'monitor': 'max dice_score'}
    #config['load_model']='C:/Users/emanu/Documents/Polito/Bioinformatics/dir/1/checkpoint-epoch1.pth'
    config['visualize']=True

    return config
