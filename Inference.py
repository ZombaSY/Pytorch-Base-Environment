from torch import device, load
from models.model import BaseNet
from torchvision.transforms import transforms
import numpy as np
from models.utils import load_cropped_image


class Inferencer:

    def __init__(self, args):
        self.file_name = args.inference_model_path
        self.src_path = args.data_path

        device('cpu')   # change device to CPU
        self.model = BaseNet(args.output_size)
        checkpoint = load(self.file_name)   # about 1 secs to load
        self.model.load_state_dict(checkpoint)

        self.trans = transforms.Compose([transforms.ToTensor(),
                                         ])
        self.result = ''

        # data transformation on grey scale image
        self.input = load_cropped_image(self.src_path, args.input_size, grey_scale=args.grey_scale, invert_color=False)
        # self.input.save('sample.jpg')

    def start_inference(self):
        input_src = self.trans(self.input)  # normalize
        input_src = input_src.unsqueeze(0)  # fix batch size to 1

        output = self.model(input_src)
        predict = output.detach().numpy()

        # classification
        self.result = str(np.argmax(predict))

        print('Result :', self.result)
