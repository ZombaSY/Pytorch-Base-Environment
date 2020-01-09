from Train import Trainer
from Inference import Inferencer
from torch.cuda import is_available


def main():
    train_mode = True
    inference_mode = False

    if train_mode:
        train_params = {'input_size': 28,
                        'output_size': 10,
                        'batch_size': 512,
                        'num_worker': 0,
                        'validation_batch_size': 1024,
                        'epoch': 5,
                        'learning_rate': 0.1,
                        'momentum': 0.5,
                        'use_cuda': True,
                        'log_interval': 10,
                        'pin_memory': True,
                        'saved_model_directory': 'model_checkpoints',
                        'train_data_path': 'A:/Users/SSY/Desktop/dataset/training',
                        'train_csv_path': 'A:/Users/SSY/Desktop/dataset/mnist_train.csv',
                        'test_data_path': 'A:/Users/SSY/Desktop/dataset/testing',
                        'test_csv_path': 'A:/Users/SSY/Desktop/dataset/mnist_test.csv'}

        print('is cuda available? :', train_params['use_cuda'] and is_available())

        trainer = Trainer(train_params)
        trainer.start_train('my_awesome_model_name')

    if inference_mode:
        inference_params = {'model_path': 'model_checkpoints/sunyonggod_1.pt',
                            'data_path': 'A:/Users/SSY/Desktop/dataset/2.jpg'}

        inferencer = Inferencer(inference_params)
        inferencer.start_inference()


if __name__ == "__main__":
    main()
