import argparse
from Training import train

"""
Main starter script for training an State-based network. Can also be used to run pure inference.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Trains an State-based network. Can also be used to run pure inference.')

    parser.add_argument('--model_save_dir', default='./models', type=str, nargs='?', help='Folder directory in which to store the trained models.')

    parser.add_argument('--data_dir', default='./datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')

    parser.add_argument('--datasets', default=[" "], nargs='+', help='The names of the datasets to use. Datasets = [Saturator, OD, ODNeutron, Pultec, FilterNeutron, LA2A, CL1B].')

    parser.add_argument('--epochs', default=60, type=int, nargs='?', help='Number of training epochs.')

    parser.add_argument('--model', default=[" "], type=int, nargs='?', help='The name of the model to train (LSTM, ED, LRU, S4D, S6).')

    parser.add_argument('--batch_size', default=8, type=int, nargs='?', help='Batch size.')

    parser.add_argument('--mini_batch_size', default=2048, type=int, nargs='?', help='Mini batch size.')

    parser.add_argument('--hidden_layer_sizes', default=8, nargs='+', help='Hidden layer sizes (amount of units) of the network.')

    parser.add_argument('--learning_rate', default=3e-4, type=float, nargs='?', help='Initial learning rate.')

    parser.add_argument('--only_inference', default=False, type=bool, nargs='?', help='When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model.')

    return parser.parse_args()


def train(args):

    print("######### Preparing for training/inference #########")
    print("\n")
    train(data_dir=args.data_dir,
          model_save_dir=args.model_save_dir,
          save_folder=f'{args.model}_{args.dataset}_{args.hidden_layer_sizes}',
          dataset=args.datasets,
          epochs=args.epochs,
          model=args.model,
          batch_size=args.batch_size,
          mini_batch_size=args.mini_batch_size,
          units=args.hidden_layer_sizes,
          learning_rate=args.learning_rate,
          inference=args.only_inference)


def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()



