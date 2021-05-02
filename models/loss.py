from .metrics import iou_score


def iou_loss(prediction, target):
    iou=iou_score(prediction, target)
    iou.requires_grad=True
    return - iou
