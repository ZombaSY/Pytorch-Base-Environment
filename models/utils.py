import numpy as np
import PIL.Image as Image
from PIL.ImageOps import invert


def cutout(*, mask_size=24, cutout_inside=False, mask_color=(255)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color

        return image

    return _cutout


def load_cropped_image(src_path, output_size, grey_scale, invert_color=False):

    def _crop_background(numpy_src):

        def _get_vertex(img):
            index = 0
            for i, items in enumerate(img):
                if items.max() != 0:  # activate where background is '0'
                    index = i
                    break

            return index

        numpy_src_y1 = _get_vertex(numpy_src)
        numpy_src_y2 = len(numpy_src) - _get_vertex(np.flip(numpy_src, 0))
        numpy_src_x1 = _get_vertex(np.transpose(numpy_src))
        numpy_src_x2 = len(numpy_src[0]) - _get_vertex(np.flip(np.transpose(numpy_src), 0))

        return numpy_src_x1, numpy_src_y1, numpy_src_x2, numpy_src_y2

    if grey_scale:
        src_image = Image.open(src_path, 'r').convert('L')
        if invert_color:
            src_image = invert(src_image)     # invert color

        numpy_image = np.asarray(src_image.getdata(), dtype=np.float64).reshape((src_image.size[1], src_image.size[0]))
        numpy_image = np.asarray(numpy_image, dtype=np.uint8)  # if values still in range 0-255

        pil_image = Image.fromarray(numpy_image, mode='L')
        x1, y1, x2, y2 = _crop_background(numpy_image)
        pil_image = pil_image.crop((x1, y1, x2, y2))
        pil_image = pil_image.resize([output_size, output_size])

    else:
        pil_image = Image.open(src_path, 'r')

    return pil_image
