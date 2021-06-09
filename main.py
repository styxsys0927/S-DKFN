import argparse
from prepare import *
from train_dkfn import *
from train_sdkfn import *

parser = argparse.ArgumentParser(description='traffic prediction')

# data
parser.add_argument('-dataset', type=str, default='PEMS08', help='choose dataset to run [options: metr_la10, metr_la20,'
                                                                 'metr_la30, metr_la40, metr_la50, PEMS08_300, PEMS08_1000,'
                                                                 'PEMS08_3000]')

# model
parser.add_argument('-model', type=str, default='sdkfn', help='choose model to train and test [options: sdkfn, dkfn]')
args = parser.parse_args()

# load data
if args.dataset == 'metr_la10':
    print("\nLoading metr_la 10 data...")
    speed_matrix = np.load('./METR_LA_Dataset/speed_matrix_10.npy')
    A = np.load('./METR_LA_Dataset/METR_LA_A.npy')
elif args.dataset == 'metr_la20':
    print("\nLoading metr_la 20 data...")
    speed_matrix = np.load('./METR_LA_Dataset/speed_matrix_20.npy')
    A = np.load('./METR_LA_Dataset/METR_LA_A.npy')
elif args.dataset == 'metr_la30':
    print("\nLoading metr_la 30 data...")
    speed_matrix = np.load('./METR_LA_Dataset/speed_matrix_30.npy')
    A = np.load('./METR_LA_Dataset/METR_LA_A.npy')
elif args.dataset == 'metr_la40':
    print("\nLoading metr_la 40 data...")
    speed_matrix = np.load('./METR_LA_Dataset/speed_matrix_40.npy')
    A = np.load('./METR_LA_Dataset/METR_LA_A.npy')
elif args.dataset == 'metr_la50':
    print("\nLoading metr_la 50 data...")
    speed_matrix = np.load('./METR_LA_Dataset/speed_matrix_50.npy')
    A = np.load('./METR_LA_Dataset/METR_LA_A.npy')
elif args.dataset == 'PEMS08_300':
    print("\nLoading PEMS08_300 data...")
    speed_matrix = np.load('PEMS08_Dataset/speed_min_300.npy')
    A = np.load('PEMS08_Dataset/pems08_adj_mat_clean.npy')
elif args.dataset == 'PEMS08_1000':
    print("\nLoading PEMS08 data...")
    speed_matrix = np.load('PEMS08_Dataset/speed_min_1000.npy')
    A = np.load('PEMS08_Dataset/pems08_adj_mat_clean.npy')
elif args.dataset == 'PEMS08_3000':
    print("\nLoading PEMS08 data...")
    speed_matrix = np.load('PEMS08_Dataset/speed_min_3000.npy')
    A = np.load('PEMS08_Dataset/pems08_adj_mat_clean.npy')
else:
    print("\nDataset not found...")
    exit(0)



# model
if args.model == 'sdkfn':
    print("\nPreparing data...")
    train_dataloader, valid_dataloader, test_dataloader, max_speed = PrepareDataset_multi(speed_matrix, args.dataset,
                                                                                          BATCH_SIZE=64,
                                                                                          pred_len=1)
    print("\nTraining sdkfn model...")
    tdkfn, tdkfn_loss = TrainsDKFN(train_dataloader, valid_dataloader, A, K=3, num_epochs=100)
    print("\nTesting sdkfn model...")
    results = TestsDKFN(tdkfn, test_dataloader, max_speed)

elif args.model == 'dkfn':
    print("\nPreparing data...")
    train_dataloader, valid_dataloader, test_dataloader, max_speed = PrepareDataset(speed_matrix, BATCH_SIZE=64,
                                                                                          pred_len=1)
    print("\nTraining dkfn model...")
    dkfn, dkfn_loss = TrainDKFN(train_dataloader, valid_dataloader, A, K=3, num_epochs=100)
    print("\nTesting dkfn model...")
    results = TestDKFN(dkfn, test_dataloader, max_speed)

else:
    print("\nModel not found...")
    exit(0)


