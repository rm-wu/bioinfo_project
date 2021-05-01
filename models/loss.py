from .metrics import iou_score


def iou_loss(prediction, target):
    return - iou_score(prediction, target)
