import torch
from tensorboardX import SummaryWriter
from abc import abstractmethod


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self,
                 model,
                 criterion,
                 #metric_fnts,
                 optimizer,
                 config):
        self.config = config

        self.model = model
        self.criterion = criterion
        #self.metric_fnts = metric_fnts
        self.optimizer = optimizer

        # TODO: make a dictionary within config['trainer'] (is a dict) with the hypParam for trainer
         # cfg_trainer = config['trainer']
        self.epochs = config['epochs']

        # TODO: save/resume checkpoint
        #self.save_period = config['save_period']
        #self.checkpoint_dir = config['save_dir']

        # TODO: Tensorboard writer
        # self.writer = Tens
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
