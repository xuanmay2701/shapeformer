import os
import argparse
import logging
# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, Initialization, Data_Loader, Data_Verifier
from Shapelet.mul_shapelet_discovery import ShapeletDiscover
import pickle

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_path', default='Dataset/UEA/', choices={'Dataset/UEA/', 'Dataset/Segmentation/'},
                    help='Data path')
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.5, help="Proportion of the train-set to be used as validation")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')

parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')

parser.add_argument("--window_size", default=100, type=float, help="window size")
parser.add_argument("--num_pip", default=0.5, type=float, help="number of pips")
parser.add_argument("--processes", default=1, type=int, help="number of processes for extracting shapelets")
parser.add_argument("--dataset_pos", default=1, type=int, help="number of processes for extracting shapelets")

parser.add_argument("--is_extract_candidate", default=1, type=int, help="is extract candidate?")
parser.add_argument("--dis_flag", default=1, type=int, help="is extract candidate?")
args = parser.parse_args()

if __name__ == '__main__':
    config = Setup(args)  # configuration dictionary
    device = Initialization(config)
    Data_Verifier(config)  # Download thex UEA and HAR datasets if they are not in the directory
    All_Results = ['Datasets', 'ConvTran']  # Use to store the accuracy of ConvTran in e.g "Result/Datasets/UEA"
    list_dataset_name = os.listdir(config['data_path'])
    list_dataset_name.sort()
    print(list_dataset_name)
    for problem in list_dataset_name[config['dataset_pos']:config['dataset_pos'] + 1]:  # for loop on the all datasets in "data_dir" directory
        print("Problem: %s" % problem)
        config['data_dir'] = config['data_path'] +"/"+ problem
        # ------------------------------------ Load Data ---------------------------------------------------------------
        logger.info("Loading Data ...")
        Data = Data_Loader(config)
        train_data = Data['train_data']
        train_label = Data['train_label']
        len_ts = Data['max_len']
        print("len_ts: %s" % len_ts)
        print("dim: %s" % train_data.shape[1])
        dim = train_data.shape[1]

        # --------------------------------------------------------------------------------------------------------------
        # -------------------------------------------- Shapelet Discovery ----------------------------------------------
        if config['is_extract_candidate']:
            shapelet_discovery = ShapeletDiscover(window_size=args.window_size, num_pip=args.num_pip,
                                                  processes=args.processes, len_of_ts=len_ts, dim=dim)
            print("extract_candidate...")
            shapelet_discovery.extract_candidate(train_data=train_data)
            sc_path = "store/" + problem + "_sd.pkl"
            file = open(sc_path, 'wb')
            pickle.dump(shapelet_discovery, file)
            file.close()
        else:
            sc_path = "store/" + problem + "_sd.pkl"
            file = open(sc_path, 'rb')
            shapelet_discovery = pickle.load(file)
            file.close()

        if args.window_size <= int(len_ts / 2):
            print("window_size:%s" % args.window_size)
            config['window_size'] = args.window_size

            sc_path = "store/" + problem + "_" + str(args.window_size) + ".pkl"
            shapelet_discovery.set_window_size(args.window_size)
            print("discovery...")
            shapelet_discovery.discovery(train_data=train_data, train_labels=train_label, flag=config['dis_flag'])
            print("save_shapelet_candidates...")
            shapelet_discovery.save_shapelet_candidates(path=sc_path)
