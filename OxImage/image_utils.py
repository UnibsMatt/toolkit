import cv2
import matplotlib.patches as ptc
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage import io
from skimage.color import rgb2gray

from OxImage.bounding_boxes_utils import *
from OxImage.bounding_boxes_utils import bb_form


def read_image_opencv(path: str):
    """
    Funzione per il caricamento di un immagine con opencv
    Args:
        path (str): Percorso dell'immagine

    Returns:
        np.ndarray uint8
    """
    img = cv2.imread(path)
    return img


def read_image_ski(path: str, channel="rgb"):
    # TODO: aggiungere funzionalità altri tipo di numpy uint16, 32 etc.
    """Funzione per la lettura di un immagine con skimage
    Args:
        path: percorso dell'immagine
        channel: ["rgb", "gray"] metodi di lettura dell'immagine

    Returns: numpy array dell'immagine uint8

    """
    assert channel in ["rgb", "gray"]
    read_as_grayscale = True if channel == "gray" else False
    image = io.imread(path).astype(np.uint8)
    if read_as_grayscale:
        image = rgb2gray(image)
    return image


def read_image_pil(path: str, resize_size=None, channel="rgb"):
    """
    Args:
        channel: [rgb, gray]-> default rgb
        path: percorso dell'immagine
        resize_size: (width, height) per fare il resize dell'immagine

    Returns: numpy array dell'immagine
    """

    assert channel in ["rgb", "gray"]
    if channel == "gray":
        image = Image.open(path, "r").convert("L")
    else:
        image = Image.open(path, "r").convert("RGB")
    if resize_size is not None:
        assert len(resize_size) == 2, "Expected tuple(width, height)"
        image = image.resize(resize_size)
    return np.asarray(image)


def __universal_image_reader(arg: (np.ndarray, str, Image.Image)):
    if isinstance(arg, np.ndarray):
        return arg
    if isinstance(arg, str):
        return read_image_pil(arg)
    if isinstance(arg, Image.Image):
        return np.array(arg)


def plot_img(
    img: (np.ndarray, str, Image.Image),
    channel=None,
    save_img=False,
    saved_img_path=".",
    show=True,
    plot_axis: bool = False,
):
    """Plot su matplotlib dell'immagine
    Args:
        plot_axis: flag to plot the axis
        show: flag if showing image during saving
        save_img: flag to save image
        saved_img_path: path to save image
        channel: channel [rgb, gray]
        img (np.ndarray): immagine
    """
    support_image = __universal_image_reader(img)
    assert channel in [None, "gray"], "Undefined channel"

    plt.figure(figsize=(10, 10))
    if not plot_axis:
        plt.axis("off")
    plt.imshow(support_image, cmap=channel)
    if save_img:
        plt.savefig(saved_img_path)
    if show:
        plt.show()
    plt.close()


def plot_img_with_bb(
    image: np.ndarray,
    bounding_boxes: [np.asarray, torch.Tensor],
    boxes_form="hw",
    labels=None,
    ground_truth_boxes=None,
    ground_truth_boxes_form="hw",
    ground_truth_labels=None,
    font_size=12,
    scale=1,
    channel=None,
    save_img=False,
    saved_img_path=".",
    show=True,
):
    """
    Plot images with bounding boxes and ground truth
    Args:
        show:
        save_img: if true save the image
        saved_img_path: path to save the image
        channel: ["gray", None] Grayscale se gray altrimenti rgb
        ground_truth_labels: label dei ground truth
        scale: multiplication scale
        boxes_form: bb form
        ground_truth_boxes_form: rappresentazione dei gt bounding boxes
        font_size: size of the font
        image: image
        bounding_boxes: array dei bounding boxes nello shape [n, 4]
        labels: labels dei bb nella forma [n, 1]
        ground_truth_boxes: bounding boxes del ground truth
    """

    assert channel in [None, "gray"], "Undefined channel"
    assert boxes_form in bb_form, f"Boxes structure undefined: use {bb_form}"
    assert ground_truth_boxes_form in bb_form, f"Ground truth boxes structure undefined: use {bb_form}"

    if labels is not None:
        assert len(labels) == len(bounding_boxes), "Boxes and label size don't match"
    if ground_truth_labels is not None:
        assert len(ground_truth_labels) == len(ground_truth_boxes), "Gt boxes and gt label size don't match"

    if ground_truth_boxes is not None:
        if isinstance(ground_truth_boxes, np.ndarray):
            gt_boxes = torch.from_numpy(ground_truth_boxes)
        else:
            gt_boxes = ground_truth_boxes
        gt_boxes = transform_bb(gt_boxes, form_in=ground_truth_boxes_form, form_out="hw") * scale
    else:
        gt_boxes = None
    if isinstance(bounding_boxes, np.ndarray):
        bboxes = torch.from_numpy(bounding_boxes)
    else:
        bboxes = bounding_boxes

    bboxes = transform_bb(bboxes, form_in=boxes_form, form_out="hw") * scale

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=channel)

    ax = plt.gca()
    """
    Per ogni bb trovato stampo il rettangolo corrisponedente.
    Se ci sono le label aggiungo le label nell'angolo a sinistra
    """
    for index, box in enumerate(bboxes):
        ax.add_patch(
            ptc.Rectangle(
                (int(box[0]), int(box[1])), int(box[2]), int(box[3]), linewidth=1, edgecolor="g", facecolor="none"
            )
        )
        if labels is not None:
            ax.text(int(box[0]), int(box[1] - 5), labels[index], fontsize=font_size, color="g")

    """
    Per ogni gt_bb stampo il rettangolo corrisponedente.
    Se ci sono le label aggiungo le label nell'angolo a sinistra
    """
    if gt_boxes is not None:
        for index, box in enumerate(gt_boxes):
            ax.add_patch(
                ptc.Rectangle(
                    (int(box[0]), int(box[1])), int(box[2]), int(box[3]), linewidth=2, edgecolor="r", facecolor="none"
                )
            )
            if ground_truth_labels is not None:
                ax.text(
                    int(box[0]), int(box[1] + box[3] + 30), ground_truth_labels[index], fontsize=font_size, color="r"
                )
    """
    Se stato impostato il flag di salvataggio salvo l'immagine nel percorso specificato da saved_img_path
    """
    if save_img:
        plt.savefig(saved_img_path)
    """Mostro e chiudo il plot"""
    if show:
        plt.show()
    plt.close()
