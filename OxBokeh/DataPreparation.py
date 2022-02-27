import base64
from io import BytesIO

import numpy as np
from PIL import Image


def to_png(image: np.ndarray):
    """Convert numpy to buffer image png
    Args:
        image: numpy array

    Returns: buffer values
    """
    image = image.astype(np.uint8)
    out = BytesIO()
    ia = Image.fromarray(image)
    ia.save(out, format="png")
    return out.getvalue()


def encode_images(images: [list, np.ndarray]):
    """Function to encode list of images/ numpy array in base 64 for bokeh visualizatoin
    Args:
        images:

    Returns:

    """
    if isinstance(images, np.ndarray):
        image_list = images.tolist()
    else:
        image_list = images
    urls = []
    for im in image_list:
        png = to_png(im)
        url = "data:image/png;base64,"
        url += base64.b64encode(png).decode("utf-8")
        urls.append(url)
    return urls
