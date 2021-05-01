import torch
# from pytorch_lightning.metrics import IoU



# TODO: usare una sola chiamata a sigmoid per entrambe le metriche

'''def iou_score(prediction, target):
    with torch.no_grad():
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()

        iou_score_total = 0
        for act, pred in zip(target, prediction):
            intersection = torch.logical_and(act, pred)
            union = torch.logical_or(act, pred)
            iou_score_total = torch.sum(intersection) / torch.sum(union)
        return iou_score_total / len(target)'''
'''
    def mean_iou_score(prediction, target, num_classes):
        Mean IoU score for 2 classes semantic segmentation
    
        :param prediction   : B x 1 x H x W
        :param target       : B x 1 x H x W
        :return iou_score   : Squeeze the batch to a 1-D tensor, then for each class compute:
                                IoU[i] = true_positive / (true_positive + false_positive + false_negative)
                              then return:
                                iou = IoU.sum() / 2 / B
                              where  2 is the number of classes (0 and 1).
'''

# TODO: Fix this score
# TODO: Versione veloce usando pytorch_lighning
def iou_score(prediction, target, threshold=0.5):
    """
    Returns the IoU for class 1 ( elemens for which(prediciton > threshold)  == 1.0)
    :param prediction:
    :param target:
    :param threshold:
    :return:
    """
    assert prediction.shape == target.shape, "Prediction shape and target shape do not match"
    N = prediction.shape[0]
    prediction = (prediction > threshold).float()

    with torch.no_grad():
        # flatten the two input vectors and compute

        # intersection(pred, target) = true_positives
        intersection = torch.logical_and(prediction.reshape(N, -1),
                                        target.reshape(N, -1))

        # union(pred, target) = true_positives + false_positives + false_negatives
        union = torch.logical_or(prediction.reshape(N, -1),
                                 target.reshape(N, -1))
        intersection_sample = intersection.sum(dim=1)       # compute the intersection for each sample
        union_sample = union.sum(dim=1)                     # compute the union for each sample
        iou_sample = (intersection_sample /
                     torch.clamp(union_sample, min=1e-15))  # compute the iou for each sample
        return torch.mean(iou_sample)                       # return the mean over all samples


def dice_score(prediction, target):
    with torch.no_grad():
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()

        dice_total = 0
        for act, pred in zip(target, prediction):
            intersection = torch.sum(torch.logical_and(act, pred))
            dice_total += 2 * intersection / (act.nelement() + pred.nelement())
        return dice_total / len(target)
