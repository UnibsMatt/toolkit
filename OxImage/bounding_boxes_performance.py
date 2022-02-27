import torch

from OxImage.image_utils import Iou


def prediction_accuracy(predicted_boxes: torch.Tensor, ground_truth_boxes: torch.Tensor):
    """
    Function to calculate accuracy given bb and ground truth
    Args:
        predicted_boxes: torch prediction bb in POINT form
        ground_truth_boxes: torch ground bb in POINT form
    """
    assert (predicted_boxes[:, :2] >= predicted_boxes[:, 2:]).all(), "BB not in point form"
    assert (ground_truth_boxes[:, :2] >= ground_truth_boxes[:, 2:]).all(), "GT BB not in point form"
    accuray = Iou(predicted_boxes, ground_truth_boxes)
    return accuray
