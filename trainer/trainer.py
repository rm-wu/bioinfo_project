from base import BaseTrainer
from utils import MetricMonitor
from tqdm import tqdm
from torchvision.transforms import CenterCrop
import torch


TILE_DIM = 400


class Trainer(BaseTrainer):
    def __init__(self,
                 model,
                 criterion,
                 #metric_fnts,
                 optimizer,
                 config,
                 train_loader,
                 device,
                 val_loader=None):
        super().__init__(model,
                         criterion,
                         #metric_fnts,
                         optimizer,
                         config)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.do_validation = self.val_loader is not None
        self.device = device
        # Crop only the central part of the tile that we want to predict
        self.center = CenterCrop(TILE_DIM)


    def _train_epoch(self, epoch):
        # TODO: epoch is for logging purposes, add this part
        metric_monitor = MetricMonitor()

        self.model.train()
        stream = tqdm(self.train_loader)

        for batch_idx, (image, mask) in enumerate(stream, start=1):
            image, mask = image.to(self.device), mask.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(self.center(output), self.center(mask))
            metric_monitor.update("Loss", loss.item())
            loss.backward()
            self.optimizer.step()
            stream.set_description(f"Epoch: {epoch} | Train\t|{metric_monitor}")

        if self.do_validation:
            self._val_epoch(epoch)

    def _val_epoch(self, epoch):
        self.model.eval()
        stream = tqdm(self.val_loader)
        metric_monitor = MetricMonitor()
        with torch.no_grad():
            for batch_idx, (image, mask) in enumerate(stream):
                image, mask = image.to(self.device), mask.to(self.device)
                output = self.model(image)
                loss = self.criterion(self.center(output), self.center(mask))
                metric_monitor.update("Loss", loss.item())
                stream.set_description(f"Epoch: {epoch} | Validation\t|{metric_monitor}")
