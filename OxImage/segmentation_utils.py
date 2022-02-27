import imgaug.augmenters as iaa
import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from OxImage.bounding_boxes_utils import *
from OxImage.image_utils import *


def crop_from_side(img: np.ndarray, crop_tuple: tuple, bounding_boxes=None, segmentation_mask=None):
    """Function used to crop the image from side
    Args:
        img: image
        crop_tuple: (top, right, bottom, left)

    Returns:
        Cropped image
    """
    assert isinstance(crop_tuple, tuple), "Expected tuple in input: top, right, bottom, left"
    seq = iaa.Sequential([iaa.Crop(px=crop_tuple, keep_size=False)])

    return seq(image=img)


def pad_to_size(img: np.ndarray, width: int, height: int, position="center"):
    """Funzione per il pad dell'immagine
    Args:
        position: posizione del pad. ["uniform", "normal", "center"] default center
        img (np.ndarray): immagine da croppare
        width (int):
        height (int):
    Returns: immagine paddata
    """
    position_options = ["uniform", "normal", "center"]
    assert position in position_options
    seq = iaa.Sequential(
        [
            iaa.PadToFixedSize(width=width, height=height, position=position),
        ]
    )
    return seq(image=img)


def crop_to_size(img: np.ndarray, width: int, height: int, cropping="center"):
    """Funzione per il crop dell'immagine
    Args:
        cropping: posizione del crop. ["uniform", "normal", "center"] default center
        img (np.ndarray): immagine da croppare
        width (int):
        height (int):
    Returns: immagine croppata

    """
    cropping_options = ["uniform", "normal", "center"]
    assert cropping in cropping_options
    seq = iaa.Sequential(
        [
            iaa.CropToFixedSize(width=width, height=height, position=cropping),
        ]
    )

    return seq(image=img)


def resize(
    img: np.ndarray,
    width: int,
    height: int,
    interpolation="cubic",
    bounding_boxes=None,
    bounding_boxes_form="hw",
    segmentation_mask=None,
):
    """funzione per il resize dell'immagine
    Args:
        bounding_boxes_form:
        bounding_boxes:
        segmentation_mask:
        interpolation: metodo di interpolazione dell'immagine. Default cubic.
        img: immagine in numpy
        width:
        height:

    Returns: immagine di dimensioni specificate

    """
    interpolation_options = ["nearest", "linear", "area", "cubic"]
    assert interpolation in interpolation_options
    seq = iaa.Sequential(
        [
            iaa.Resize({"height": height, "width": width}, interpolation=interpolation),
        ]
    )
    if bounding_boxes is not None:
        bb = transform_bb(bounding_boxes, form_in=bounding_boxes_form, form_out="point")
    else:
        bb = None
    # IMPORTANTE: al sequencer bisogna passare i bb in forma point!
    return sequencer(seq, img, bb, segmentation_mask)


def get_mask_from_bb(
    image_shape: [tuple, list, np.ndarray],
    bounding_boxes: np.ndarray,
    bounding_boxes_form="hw",
    return_aug_seg_map=False,
):
    """
    Restituisce la maschera a partire dallo shape dell'immagine e dai bounding boxes.

    Args:
        image_shape: tupla width height
        bounding_boxes: bounding boxes
        bounding_boxes_form: forma dei bb
        return_aug_seg_map: flag per ritornare la maschera di segmentazione di Iaa o numpy array

    Returns:
        maschera nella forma specificata

    """
    mask = np.zeros(image_shape[:2], dtype=bool)
    # trasformo i bb nella point form
    bounding_boxes = transform_bb(bounding_boxes, form_in=bounding_boxes_form, form_out="point")
    for bounding_box in bounding_boxes:
        mask[int(bounding_box[1]) : int(bounding_box[3]), int(bounding_box[0]) : int(bounding_box[2])] = 1
    if return_aug_seg_map:
        return SegmentationMapsOnImage(mask, image_shape)
    return mask


def get_bounding_boxes_on_image(bounding_boxes: [np.ndarray, torch.Tensor], image_shape: tuple, bounding_box_form="hw"):
    """
    Funzione per ricevere i bb in formato iaa
    Args:
        bounding_boxes:
        image_shape:
        bounding_box_form:

    Returns:

    """
    assert bounding_box_form in bb_form, f"Unexpected bb form. Accepted: {bb_form}"
    assert isinstance(image_shape, tuple)
    assert len(image_shape) >= 2, f"Image should have 2 or 3 dimension, found {len(image_shape)}"
    if isinstance(bounding_boxes, np.ndarray):
        support_bb = torch.from_numpy(bounding_boxes)
    else:
        support_bb = bounding_boxes
    bbs = BoundingBoxesOnImage(
        [BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]) for bb in support_bb], shape=image_shape
    )
    return bbs


def bounding_box_on_images_to_numpy(bounding_boxes: BoundingBoxesOnImage):
    """Funzione per la trasformazione da bb iaa in bb numpy

    Args:
        bounding_boxes: bounding boxes in forma iaa

    Returns:
        bounding boxes in point form of numpy

    """
    bb_array = []
    assert isinstance(bounding_boxes, BoundingBoxesOnImage)

    for bb in bounding_boxes.items:
        bb_array.append(bb.coords.flatten())
    return np.asarray(bb_array)


def sequencer(
    sequential: iaa.Sequential,
    image: np.ndarray,
    bounding_boxes: [np.ndarray, torch.Tensor, None],
    segmentation_mask: [np.ndarray, None],
):
    """
    Sequencer for image bb and mask
        Args:
            sequential: the iaa.Sequential module
            image: numpy image
            bounding_boxes: numpy bb in point form
            segmentation_mask: numpy array of mask

        Returns:
            Depending on the input parameters, img bb and mask augmented as numpy array

    """
    bbs = None
    seg_mask = None
    # se ci sono i bb allora li trasformo in bbonimage di iaa
    if bounding_boxes is not None:
        bbs = get_bounding_boxes_on_image(bounding_boxes, image.shape[:2])

    # se c'è la maschera la trasformo segmentation maps on image di iaa
    if segmentation_mask is not None:
        seg_mask = SegmentationMapsOnImage(segmentation_mask, image.shape[:2])
    image, bbs, seg_mask = sequential(image=image, bounding_boxes=bbs, segmentation_maps=seg_mask)

    # altra possibilità, ho la maschera ma non i bb
    if bbs is None:
        if seg_mask is None:
            return image
        return image, np.squeeze(seg_mask.arr, 2)
    if seg_mask is None:
        return image, bounding_box_on_images_to_numpy(bbs)
    return image, bounding_box_on_images_to_numpy(bbs), np.squeeze(seg_mask.arr, 2)
