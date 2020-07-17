from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import pandas as pd
from PIL.Image import open, new


class ImageCSVLoader(Dataset):

    def __init__(self, transform, train_data_path, train_label_path, is_grey_scale):
        self.transform = transform
        self.is_grey_scale = is_grey_scale

        x_img_name = os.listdir(train_data_path)
        y_label = pd.read_csv(train_label_path, header=0)
        y_label = y_label['label']  # label column

        x_img_path = list()
        for item in x_img_name:
            x_img_path.append(train_data_path + '/' + item)

        self.len = len(x_img_name)
        self.x_img_path = x_img_path
        self.y_label = y_label

    def __getitem__(self, index):
        new_img = open(self.x_img_path[index])

        if not self.is_grey_scale:
            rgb_img = new("RGB", new_img.size)
            rgb_img.paste(new_img)

        out_img = self.transform(new_img)

        return out_img, self.y_label[index]     # data, target

    def __len__(self):
        return self.len


class ValidationLoader:

    def __init__(self, dataset_path, label_path, input_size, is_grey_scale, batch_size=64, num_workers=0, pin_memory=True):
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.validation_data_path = dataset_path
        self.validation_label_path = label_path
        self.is_grey_scale = is_grey_scale

        # Data augmentation and normalization
        self.validation_trans = transforms.Compose([transforms.Resize(self.input_size),
                                                    transforms.ToTensor(),
                                                    ])

        self.ValidationDataLoader = DataLoader(ImageCSVLoader(self.validation_trans,
                                                              self.validation_data_path,
                                                              self.validation_label_path,
                                                              self.is_grey_scale),
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=pin_memory)

    def __len__(self):
        return self.ValidationDataLoader.__len__()


class TrainLoader:

    def __init__(self, dataset_path, label_path, input_size, is_grey_scale, batch_size=64, num_workers=0, pin_memory=True):
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_data_path = dataset_path
        self.train_label_path = label_path
        self.is_grey_scale = is_grey_scale

        # # Data augmentation and normalization
        self.train_trans = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomRotation(30),
                                               transforms.ColorJitter(),
                                               transforms.Resize(self.input_size),
                                               transforms.ToTensor(),
                                               ])

        self.TrainDataLoader = DataLoader(ImageCSVLoader(self.train_trans,
                                                         self.train_data_path,
                                                         self.train_label_path,
                                                         self.is_grey_scale),
                                          batch_size=self.batch_size,
                                          num_workers=self.num_workers,
                                          shuffle=True,
                                          pin_memory=self.pin_memory)

    def __len__(self):
        return self.TrainDataLoader.__len__()
