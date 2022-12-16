import torch
from model import *
from loss_function import *
from utility import *
import h5py


if __name__ == "__main__":
    args = parse()

    h_io = h5py.File(args.h5_file, 'r')
    net = Vnet(1, 7)   # 7 classification
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    for i in range(args.num_epochs):
        train_ds = getData(h_io, args.num_images, args.num_clips)  # 64, 32
        test_ds = getData(h_io, args.num_images // 8, args.num_clips, single=args.single)
        train_iter = load_data(train_ds, args.batch_size)
        test_iter = load_data(test_ds, args.batch_size)

        print("epoch:", epoch + 1)

        train(net, train_iter, test_iter, PCL, FDL, optimizer, args.dict_dir, args.num_loops)

        ckpt = torch.load(args.dict_dir)
        net.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])