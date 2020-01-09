from torch import device, load
import PIL.Image as Image
from PIL.ImageOps import invert
from models.model import BaseNet
from torchvision.transforms import transforms
import numpy as np


class Inferencer:

    def __init__(self, parameters):
        self.file_name = parameters['model_path']
        self.src_path = parameters['data_path']

        device('cpu')  # change device to CPU
        self.model = BaseNet(10)
        # about 1 secs to load
        checkpoint = load(self.file_name)
        self.model.load_state_dict(checkpoint)

        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

        self.input = self.convert_handmade_src(self.src_path, 28)
        self.result = ''

        self.input.save('sample.jpg')  #@@@@@@@@@@@@@@@@

    def start_inference(self):
        input_src = self.trans(self.input)  # normalize
        input_src = input_src.unsqueeze(0)  # fix batchsize to 1

        output = self.model(input_src)
        predict = output.detach().numpy()

        self.result = str(np.argmax(predict))

        print('Result :', self.result)

    @staticmethod
    def convert_handmade_src(src_path, output_size):

        def crop_background(numpy_src):

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

        src_image = Image.open(src_path, 'r').convert('L')
        src_image = invert(src_image)     # Cause MNIST data is inverted

        numpy_image = np.asarray(src_image.getdata(), dtype=np.float64).reshape((src_image.size[1], src_image.size[0]))
        numpy_image = np.asarray(numpy_image, dtype=np.uint8)  # if values still in range 0-255

        pil_image = Image.fromarray(numpy_image, mode='L')
        x1, y1, x2, y2 = crop_background(numpy_image)
        pil_image = pil_image.crop((x1, y1, x2, y2))
        pil_image = pil_image.resize([output_size, output_size])

        return pil_image