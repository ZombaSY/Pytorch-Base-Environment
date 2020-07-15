from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.model import BaseNet
from models.dataloader import TrainLoader, ValidationLoader
import models.lr_scheduler as lr_scheduler
import time
import os
import wandb


class Classifier:
    def __init__(self, args):
        self.startime = time.time()
        self.args = args

        # Check Cuda available and assign to device
        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # 'Init' means that this variable must be initialized.
        # 'Set' means that this variable have chance of being set, not must.
        if not self.args.skip_validation:
            self.train_loader, self.validation_loader = self.__init_data_loader()
        else:
            self.train_loader = self.__init_data_loader()
        self.model = self.__init_model()
        self.optimizer = self.__init_optimizer()
        self.scheduler = self.set_scheduler()

        if self.args.use_wandb:
            wandb.init(project='undefined-project', config=self.args)
            wandb.watch(self.model)

    def __init_data_loader(self):
        train_loader = None
        validation_loader = None

        # pin_memory = use CPU on data loader during GPU is training
        train_loader = TrainLoader(dataset_path=self.args.train_data_path,
                                   label_path=self.args.train_csv_path,
                                   input_size=self.args.input_size,
                                   batch_size=self.args.batch_size,
                                   num_workers=self.args.worker,
                                   pin_memory=self.args.pin_memory,
                                   is_grey_scale=self.args.grey_scale)

        if not self.args.skip_validation:
            validation_loader = ValidationLoader(dataset_path=self.args.test_data_path,
                                                 label_path=self.args.test_csv_path,
                                                 input_size=self.args.input_size,
                                                 batch_size=self.args.batch_size,
                                                 num_workers=self.args.worker,
                                                 pin_memory=self.args.pin_memory,
                                                 is_grey_scale=self.args.grey_scale)

            return train_loader.TrainDataLoader, validation_loader.ValidationDataLoader
        else:
            return train_loader.TrainDataLoader

    def __init_model(self):

        # Add some Nets...
        model = BaseNet(self.args.output_size).to(self.device)

        return model

    def __init_optimizer(self):
        # Add some Optimizers...
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)

        return optimizer

    def __train(self, model, device, train_loader, optimizer, epoch, scheduler=None):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            # using 'nll_loss with log scaled value' results the same effect of one hot labeled set
            target = target.long()
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            # assign changed lr
            if scheduler is not None:
                scheduler.step()

            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

                if self.args.use_wandb:
                    wandb.log({"Test_Loss": loss})

    def __validate(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # compute loss
                target = target.long()
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        # Acc can not be applied on regression model.
        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        if self.args.use_wandb:
            wandb.log({"Accuracy": 100. * correct / len(test_loader.dataset)})

    def set_scheduler(self):

        scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=self.optimizer,
                                                      warmup_steps=1,
                                                      t_total=self.train_loader.__len__(),
                                                      cycles=0.5,
                                                      last_epoch=-1)

        return scheduler

    def start_train(self, model_name):

        if not os.path.exists(self.args.saved_model_directory):
            os.mkdir(self.args.saved_model_directory)

        for epoch in range(1, self.args.epoch + 1):
            print()
            self.__train(self.model, self.device, self.train_loader, self.optimizer, epoch, self.scheduler)
            if not self.args.skip_validation:
                self.__validate(self.model, self.device, self.validation_loader)

            print("{} epoch elapsed time : {}".format(epoch, time.time() - self.startime))

            if epoch % self.args.save_interval == 0:
                self.save_model(model_name, self.args.saved_model_directory, epoch)

    def save_model(self, model_name, add_path, epoch):
        print('saving model...')
        # normal location
        file_path = add_path + '/'

        # wandb local location
        # file_path = os.path.join(wandb.run.dir, add_path + '/')

        file_format = file_path + model_name + "_" + str(epoch) + ".pt"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        file_name = file_format
        torch.save(self.model.state_dict(), file_name)
        print("{} Model Saved!!".format(file_name))
