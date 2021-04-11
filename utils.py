from collections import defaultdict
import torch

def iou(actual, predicted):
    iou_score_total=0
    for act, pred in zip(actual, predicted):
        intersection=torch.logical_and(act, pred)
        union=torch.logical_or(act, pred)
        iou_score_total=torch.sum(intersection)/torch.sum(union)
    return iou_score_total/len(actual)

def dice(actual, predicted):
    dice_total = 0
    for act, pred in zip(actual, predicted):
        intersection = torch.sum(torch.logical_and(act, pred))
        dice_total+=2*intersection/(act.nelement()+pred.nelement())
    return dice_total/len(actual)

class MetricMonitor:
    """
    Class used to monitor the evolution of different metrics for the model
    """
    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric['val'] += val
        metric['count'] += 1
        metric['avg'] = metric['val'] / metric['count']

    def return_value(self, metric_name):
        metric = self.metrics[metric_name]
        return metric['avg']

    def __str__(self):
        return " | ".join(
            [
                f"{metric_name}: {metric['avg']:.{self.float_precision}f}"
                for (metric_name, metric) in self.metrics.items()
            ]
        )
