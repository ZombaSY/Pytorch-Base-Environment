from PIL.Image import open, fromarray
from PIL.ImageOps import invert
import numpy as np


def crop_background(numpy_src):

    def _get_vertex(img):
        index = 0
        for i, items in enumerate(img):
            if items.max() != 0:    # activate where background is 0
                index = i
                break

        return index

    numpy_src_y1 = _get_vertex(numpy_src)
    numpy_src_y2 = len(numpy_src) - _get_vertex(np.flip(numpy_src, 0))
    numpy_src_x1 = _get_vertex(np.transpose(numpy_src))
    numpy_src_x2 = len(numpy_src[0]) - _get_vertex(np.flip(np.transpose(numpy_src), 0))

    return numpy_src_x1, numpy_src_y1, numpy_src_x2, numpy_src_y2

src_path = 'A:/Users/SSY/Desktop/dataset/0.jpg'

src = open(src_path, 'r').convert('L')
src = invert(src)

numpy_y = np.asarray(src.getdata(), dtype=np.float64).reshape((src.size[1], src.size[0]))
numpy_y = np.asarray(numpy_y, dtype=np.uint8)   # if values still in range 0-255

w = fromarray(numpy_y, mode='L')
x1, y1, x2, y2 = crop_background(numpy_y)
w = w.crop((x1, y1, x2, y2))
w = w.resize([28, 28])

w.save('out.jpg')
