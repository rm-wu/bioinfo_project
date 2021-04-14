import torch

# TODO: usare una sola chiamata a sigmoid per entrambe le metriche

def iou_score(prediction, target):
    with torch.no_grad():
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()

        iou_score_total = 0
        for act, pred in zip(target, prediction):
            intersection = torch.logical_and(act, pred)
            union = torch.logical_or(act, pred)
            iou_score_total = torch.sum(intersection) / torch.sum(union)
        return iou_score_total / len(target)


def dice_score(prediction, target):
    with torch.no_grad():
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()

        dice_total = 0
        for act, pred in zip(target, prediction):
            intersection = torch.sum(torch.logical_and(act, pred))
            dice_total += 2 * intersection / (act.nelement() + pred.nelement())
        return dice_total / len(target)
