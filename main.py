from train_classifier import Classifier
from Inference import Inferencer
from torch.cuda import is_available
import argparse


def main():
    parser = argparse.ArgumentParser()

    # Environment argument
    parser.add_argument('--mode', choices=['train_classifier', 'train_ae', 'inference'], help='run mode')
    parser.add_argument('--cuda', action='store_true', help='Using GPU processor')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval per batch')
    parser.add_argument('--pin_memory', action='store_true', help='Load dataset while learning')
    parser.add_argument('--save_interval', type=int, default=5, help='Saving model interval to epoch')
    parser.add_argument('--use_wandb', action='store_true', help='utilize wandb')

    # Train parameter
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--worker', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='Learning rate momentum')
    parser.add_argument('--input_size', type=int, default=228)
    parser.add_argument('--output_size', type=int, default=1)
    parser.add_argument('--grey_scale', action='store_true', help='on grey scale image')
    parser.add_argument('--skip_validation', action='store_true', help='skip validation')

    # Data parameter
    parser.add_argument('--saved_model_directory', type=str, default='model_checkpoints')
    parser.add_argument('--saved_model_name', type=str, default='model')
    parser.add_argument('--train_data_path', type=str, default='A:/Users/SSY/Desktop/dataset/MNIST/train')
    parser.add_argument('--train_csv_path', type=str, default='A:/Users/SSY/Desktop/dataset/MNIST/mnist_train.csv')
    parser.add_argument('--test_data_path', type=str, default='A:/Users/SSY/Desktop/dataset/MNIST/test')
    parser.add_argument('--test_csv_path', type=str, default='A:/Users/SSY/Desktop/dataset/MNIST/mnist_test.csv')

    # Inference parameter
    parser.add_argument('--inference_model_path', type=str, default='model_checkpoints/model_10.pt')
    parser.add_argument('--data_path', type=str, default='A:/Users/SSY/Desktop/dataset/MNIST/test/00000.jpg')

    args = parser.parse_args()

    if args.mode == 'train_classifier':
        print('Use CUDA :', args.cuda and is_available())

        classifier = Classifier(args)
        classifier.start_train(model_name=args.saved_model_name)

    elif args.mode == 'train_ae':
        pass
    
    elif args.mode == 'inference':
        inferencer = Inferencer(args)
        inferencer.start_inference()

    else:
        raise Exception('Please choose \'--mode\' in \"train_classifier, train_ae, inference \"')


if __name__ == "__main__":
    main()
