from math import inf

import torch
from tensorboardX import SummaryWriter
from abc import abstractmethod
#from utils import update
import os
from utils import MetricMonitor

class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self,
                 model,
                 criterion,
                 metrics,
                 optimizer,
                 lr_scheduler,
                 config, ):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        cfg_trainer = config['trainer']
        self.epochs = config['epochs']
        #self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # TODO : checkpoint handling
        # self.checkpoint_dir = config['save_dir']
        #self.tensorboard_dir=config['tensorboard_dir']

        # id_folder = int(max(os.listdir(self.checkpoint_dir), default=0)) + 1
        # print(id_folder)

        # self.checkpoint_dir = self.checkpoint_dir + f'/{id_folder}'
        # os.mkdir(self.checkpoint_dir)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # TODO: make a dictionary within config['trainer'] (is a dict) with the hypParam for trainer
        # cfg_trainer = config['trainer']
        self.epochs = config['epochs']

        # TODO: save/resume checkpoint
        # self.save_period = config['save_period']
        # self.checkpoint_dir = config['save_dir']
        # TODO: Tensorboard writer
        self.writer = SummaryWriter()
        '''
        # TODO: Resume Network
        if config.resume is not None:
        '''

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Train logic for an epoch

        :param epoch: Current epoch number
        :return: None
        """
        raise NotImplemented

    def train(self):
        for epoch in range(self.config['epochs']):
            result = self._train_epoch(epoch)

            # TODO:     1) log the results
            #           2) save_checkpoint
            #           3) resume_from_checkpoint

            log = {'epoch': epoch}
            log[self.mnt_metric]=result

            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:

                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    print('IMPROVED')
                    self.mnt_best = log[self.mnt_metric]
                    best = True
                else:
                    print('NOT IMPROVED')

            # self._save_checkpoint(epoch, save_best=best)

    '''
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir + '/checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)

        if save_best:
            best_path = str(self.checkpoint_dir + '/model_best.pth')
            torch.save(state, best_path)
            print('saving current best: model_best.pth')
    '''