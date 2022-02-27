import numpy as np
import torch
from torchvision.ops.boxes import box_area

bb_form = ["center", "point", "hw"]


def transform_bb(bounding_box: [np.ndarray, torch.Tensor], form_in="center", form_out="point"):
    """Function to transform bounding box from one representation to another
    Args:
        bounding_box: numpy array or Tensor of dimension [n, 4]
        form_in: ["center", "point", "hw"]
        form_out: ["center", "point", "hw"]
    """
    assert form_in in operation_in, "Operation allowed are center, point, hw"
    assert form_out in operation_in, "Operation allowed are center, point, hw"
    assert len(bounding_box.shape) == 2, "Shape must be 2. Try [[x, y, h, w]]"
    # se ho un numpy array lo trasformo in tensore
    if isinstance(bounding_box, np.ndarray):
        bounding_box = torch.from_numpy(bounding_box)
    if isinstance(bounding_box, torch.Tensor):
        pass
    transformation_in = operation_in.get(form_in)
    transformation_out = operation_out.get(form_out)
    bb = torch.clone(bounding_box)

    return transformation_out(transformation_in(bb))


def __transform_to_point_from_center(bounding_boxes) -> torch.Tensor:
    """
    Tranform bb to point x1,y1,x2,y2 to center
    Args:
        bounding_boxes:

    Returns:
        torch tensor
    """
    bounding_boxes[:, :2] = bounding_boxes[:, :2] - bounding_boxes[:, 2:] / 2
    bounding_boxes[:, 2:] = bounding_boxes[:, :2] + bounding_boxes[:, 2:]
    return bounding_boxes


def __no_op(bounding_boxes) -> torch.Tensor:
    """
    Do no operation
    Args:
        bounding_boxes:

    Returns:
        torch.Tensor
    """
    return bounding_boxes


def __transform_to_point_from_hw(bounding_boxes) -> torch.Tensor:
    """
    Transformation from point to h w
    Args:
        bounding_boxes:

    Returns:
        torch.Tensor
    """
    bounding_boxes[:, 2:] = bounding_boxes[:, :2] + bounding_boxes[:, 2:]
    return bounding_boxes


def __transform_to_center_from_point(bounding_boxes) -> torch.Tensor:
    """
    Transformation from centerx1, centery1, w, h to point
    Args:
        bounding_boxes:

    Returns:

    """
    bounding_boxes[:, 2:] = bounding_boxes[:, 2:] - bounding_boxes[:, :2]
    bounding_boxes[:, :2] = bounding_boxes[:, :2] + bounding_boxes[:, 2:] / 2
    return bounding_boxes


def __transform_to_hw_from_point(bounding_boxes: torch.Tensor) -> torch.Tensor:
    """
    Transformation from x1, y1, h, w to point
    Args:
        bounding_boxes:

    Returns:
        Tensor
    """
    bounding_boxes[:, 2:] = bounding_boxes[:, 2:] - bounding_boxes[:, :2]
    return bounding_boxes


def intersect(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Function used to calculate the intersection of the boxes
    The boxes are in form x0y0x1y1
    Args:
        boxes1: tensor [N, 4]
        boxes2: tensor [m, 4]
    """
    top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    return inter


def Iou(boxes1: torch.Tensor, boxes2: torch.Tensor, return_union=False):
    """Function used to calculate the Iou of the boxes
    The boxes are in form x0y0x1y1
        Args:
            boxes1: tensor [N, 4]
            boxes2: tensor [m, 4]
    """
    inter = intersect(boxes1, boxes2)

    area1 = box_area(boxes1)  # [N, 4]
    area2 = box_area(boxes2)  # [M, 4]
    a = area1[:, None] + area2
    union = area1[:, None] + area2 - inter
    if return_union:
        return inter / union, union
    return inter / union


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def G_Iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Function used to calculate the GIou of the boxes. Generalized IoU from https://giou.stanford.edu/
    The boxes are in form x0y0x1y1
        Args:
            boxes1: tensor [N, 4]
            boxes2: tensor [m, 4]
        Returns: float loss
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def D_Iou(bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> torch.Tensor:
    """
    D-Intersection over union
    Args:
        bboxes1: bounding boxes predicted
        bboxes2: bounding boxes expected

    Returns:
        torch.Tensor
    """

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - inter_diag / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious


operation_in = {
    "center": __transform_to_point_from_center,
    "point": __no_op,
    "hw": __transform_to_point_from_hw,
}

operation_out = {
    "center": __transform_to_center_from_point,
    "point": __no_op,
    "hw": __transform_to_hw_from_point,
}
