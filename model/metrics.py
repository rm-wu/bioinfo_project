import torch


def iou_score(prediction, target):
    with torch.no_grad():
        iou_score_total = 0
        for act, pred in zip(target, prediction):
            intersection = torch.logical_and(act, pred)
            union = torch.logical_or(act, pred)
            iou_score_total = torch.sum(intersection) / torch.sum(union)
        return iou_score_total / len(target)


def dice_score(prediction, target):
    with torch.no_grad():
        dice_total = 0
        for act, pred in zip(target, prediction):
            intersection = torch.sum(torch.logical_and(act, pred))
            dice_total += 2 * intersection / (act.nelement() + pred.nelement())
        return dice_total / len(target)
