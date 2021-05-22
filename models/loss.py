import torch

from .metrics import iou_score


def iou_loss_(prediction, target):
    return - iou_score(prediction, target)

def iou_loss(prediction, target, threshold=0.5):
    assert prediction.shape == target.shape, "Prediction shape and target shape do not match"
    N = prediction.shape[0]
    prediction = (prediction > threshold).float()

    # flatten the two input vectors and compute

    # intersection(pred, target) = true_positives
    intersection = torch.logical_and(prediction.reshape(N, -1),
                                     target.reshape(N, -1))

    # union(pred, target) = true_positives + false_positives + false_negatives
    union = torch.logical_or(prediction.reshape(N, -1),
                             target.reshape(N, -1))
    intersection_sample = intersection.sum(dim=1)  # compute the intersection for each sample
    union_sample = union.sum(dim=1)  # compute the union for each sample
    iou_sample = (intersection_sample /
                  torch.clamp(union_sample, min=1e-15))  # compute the iou for each sample
    result = -torch.mean(iou_sample)
    result.requires_grad = True
    return result