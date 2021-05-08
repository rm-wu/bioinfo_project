from base import BaseTrainer
from utils import MetricMonitor
from tqdm import tqdm
from torchvision.transforms import CenterCrop
import torch
import torchvision.transforms.functional as TF
from PIL import Image

TILE_DIM = 400


class Trainer(BaseTrainer):
    def __init__(self,
                 model,
                 criterion,
                 metrics,
                 optimizer,
                 lr_scheduler,
                 config,
                 train_loader,
                 device,
                 val_loader=None):
        super().__init__(model,
                         criterion,
                         metrics,
                         optimizer,
                         lr_scheduler,
                         config)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.do_validation = self.val_loader is not None
        self.device = device
        # Crop only the central part of the tile that we want to predict
        self.center = CenterCrop(TILE_DIM)

    def visualize_image_batch(self, output, mask):
        for idx in range(output.size()[0]):
            mask_size = mask[idx].size()
            size_out = output[idx].size()
            mask1 = torch.cat([torch.zeros(2, mask_size[1], mask_size[2]), mask[idx]])
            print(mask1.size())
            heatmap = torch.cat([output[idx], torch.zeros(2, size_out[1], size_out[2])])
            print(heatmap.size())
            heatmap = TF.to_pil_image(self.center(heatmap))
            mask_idx = TF.to_pil_image(self.center(mask1))
            res = Image.blend(mask_idx, heatmap, 0.5)
            res.show()

    def _train_epoch(self, epoch):
        # TODO: epoch is for logging purposes, add this part
        metric_monitor = MetricMonitor()

        self.model.train()
        stream = tqdm(self.train_loader)

        for batch_idx, (image, mask) in enumerate(stream, start=1):
            image, mask = image.to(self.device), mask.to(self.device)
            print(image.size())
            self.optimizer.zero_grad()
            output = self.model(image)

            if self.visualize==True and batch_idx==1:
                self.visualize_image_batch(output, mask)

            loss = self.criterion(self.center(output), self.center(mask))
            metric_monitor.update("Loss", loss.item())

            loss.backward()

            for metric in self.metrics:
                metric_monitor.update(metric.__name__,
                                      metric(self.center(output),
                                             self.center(mask)))

            self.optimizer.step()
            stream.set_description(f"Epoch: {epoch} | Train\t|{metric_monitor}")

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.writer.add_scalar("Loss_Training", metric_monitor.return_value('Loss'), epoch)
        for metric in self.metrics:
            self.writer.add_scalar(f"{metric.__name__}_Training",
                                   metric_monitor.return_value(metric.__name__), epoch)

        if self.do_validation:
            return self._val_epoch(epoch)

    def _val_epoch(self, epoch):
        self.model.eval()
        stream = tqdm(self.val_loader)
        metric_monitor = MetricMonitor()
        with torch.no_grad():
            for batch_idx, (image, mask) in enumerate(stream):

                image, mask = image.to(self.device), mask.to(self.device)
                output = self.model(image)

                loss = self.criterion(self.center(output), self.center(mask))
                #print(loss.item())
                metric_monitor.update("Loss", loss.item())
                for metric in self.metrics:
                    metric_monitor.update(metric.__name__,   metric(self.center(output),
                                                 self.center(mask)))
                stream.set_description(f"Epoch: {epoch} | Validation\t|{metric_monitor}")

        self.writer.add_scalar("Loss_Validation", metric_monitor.return_value('Loss'), epoch)
        for metric in self.metrics:
            self.writer.add_scalar(f"{metric.__name__}_Validation",
                                   metric_monitor.return_value(metric.__name__), epoch)
        return metric_monitor.return_value(str(self.mnt_metric))