from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.model import BaseNet
from models.dataloader import TrainLoader, ValidationLoader
import models.lr_scheduler as lr_scheduler
import time
import os


def _train(parameters, model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # using both 'logsoftmax' and 'nllloss' results in the same effect of one hot labeled set
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if batch_idx % parameters['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            with open(parameters['saved_model_directory'] + '/Log.txt', 'a') as f:
                f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def _validate(parameters, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    with open(parameters['saved_model_directory'] + '/Log.txt', 'a') as f:
        f.write('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


class Trainer:
    def __init__(self, parameters):
        self.startime = time.time()
        self.parameters = parameters

        # Check Cuda available and assign to device
        use_cuda = self.parameters['use_cuda'] and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.kwargs = {'num_workers': self.parameters['num_worker'],
                       'pin_memory': self.parameters['pin_memory']} if use_cuda else {}

        # 'Init' means that this variable must be initialized.
        # 'Set' means that this variable have chance of being set, not must.
        self.train_loader, self.validation_loader = self.__init_dataloader(kwargs=self.kwargs)
        self.model = self.__init_model()
        self.optimizer = self.__init_optimizer()
        self.scheduler = self.set_scheduler()

    def __init_dataloader(self, kwargs):
        # pin_memory = use CPU on dataloader during GPU is training
        train_loader = TrainLoader(dataset_path=self.parameters['train_data_path'],
                                   label_path=self.parameters['train_csv_path'],
                                   input_size=self.parameters['input_size'],
                                   batch_size=self.parameters['batch_size'],
                                   **kwargs)

        validation_loader = ValidationLoader(dataset_path=self.parameters['test_data_path'],
                                             label_path=self.parameters['test_csv_path'],
                                             input_size=self.parameters['input_size'],
                                             batch_size=self.parameters['batch_size'],
                                             **kwargs)

        return train_loader.TrainDataLoader, validation_loader.ValidationDataLoader

    def __init_model(self):
        # Add some Nets...
        model = BaseNet(self.parameters['output_size']).to(self.device)

        return model

    def __init_optimizer(self):
        # Add some Optimizers...
        optimizer = optim.SGD(self.model.parameters(), lr=self.parameters['learning_rate'])

        return optimizer

    def set_scheduler(self):
        # Add some Schedulers...
        scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=self.optimizer, warmup_steps=1, t_total=60000,
                                                      cycles=0.5, last_epoch=-1)

        return scheduler

    def start_train(self, model_name):
        #  increase the epochs if user has low amount of data
        prev_model_name = ''

        if not os.path.exists(self.parameters['saved_model_directory']):
            os.mkdir(self.parameters['saved_model_directory'])

        with open(self.parameters['saved_model_directory'] + '/Log.txt', 'w') as f:
            f.write('')
    
            for epoch in range(1, self.parameters['epoch'] + 1):
                if epoch == 1:
                    _train(self.parameters, self.model, self.device, self.train_loader, self.optimizer, epoch, self.scheduler)
                    _validate(self.parameters, self.model, self.device, self.validation_loader)
                else:
                    prev_file_name = torch.load(prev_model_name)
                    self.model.load_state_dict(prev_file_name)
                    _train(self.parameters, self.model, self.device, self.train_loader, self.optimizer, epoch, self.scheduler)
                    _validate(self.parameters, self.model, self.device, self.validation_loader)

                print("{} epoch elapsed time : {}".format(epoch, time.time() - self.startime))
                f.write("{} epoch elapsed time : {}\n".format(epoch, time.time() - self.startime))
                prev_model_name = self.save_model(self.parameters['saved_model_directory'] + '/' + model_name, epoch)

    def save_model(self, model_name, epoch):
        file_name = model_name + "_" + str(epoch) + ".pt"
        torch.save(self.model.state_dict(), file_name)
        print("{} Model Saved!!\n".format(file_name))

        return file_name
