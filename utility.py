import random
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import Counter
import time
from tqdm import tqdm


def get_list(num_elements, h5_file):
    image_list = ["patient/" + name + "/series/0/img" for name in h5_file["patient"]]
    label_list = ["patient/" + name + "/series/0/seg" for name in h5_file["patient"]]
    seq = list(range(151))
    random.shuffle(seq)
    image_list = [image_list[i] for i in seq[:num_elements]]
    label_list = [label_list[i] for i in seq[:num_elements]]

    return image_list, label_list


class RandomUnet3DData(Dataset):
    def __init__(self, h5_file, image_list, label_list):
        self.h_io = h5_file
        self.image_list = image_list
        self.label_list = label_list
        self.h = random.randint(32, 448)
        self.w = random.randint(32, 448)
        self.d = random.randint(0, 128)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        label_name = self.label_list[index]

        image_array = np.transpose(np.array(self.h_io[image_name]), (2, 0, 1))
        label_array = np.transpose(np.array(self.h_io[label_name]), (2, 0, 1))

        image_array = image_array.astype(np.float32)

        image_array = torch.FloatTensor(image_array)[self.d:self.d + 32, self.h:self.h + 64,
                      self.w:self.w + 64].unsqueeze(0)  # [1, 16, 32, 32]
        label_array = torch.LongTensor(label_array)[self.d:self.d + 32, self.h:self.h + 64,
                      self.w:self.w + 64]  # [16, 32, 32]

        return image_array, label_array

    def __len__(self):
        return len(self.image_list)


class ConstrainedUnet3DData(Dataset):
    def __init__(self, h5_file, image_list, label_list):
        self.h_io = h5_file
        self.image_list = image_list
        self.label_list = label_list
        self.h = random.randint(96, 304)  # clip
        self.w = random.randint(96, 304)
        self.d = random.randint(24, 96)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        label_name = self.label_list[index]

        image_array = np.transpose(np.array(self.h_io[image_name]), (2, 0, 1))
        label_array = np.transpose(np.array(self.h_io[label_name]), (2, 0, 1))

        image_array = image_array.astype(np.float32)

        image_array = torch.FloatTensor(image_array)[self.d:self.d + 32, self.h:self.h + 64,
                      self.w:self.w + 64].unsqueeze(0)  # [1, 16, 32, 32]
        label_array = torch.LongTensor(label_array)[self.d:self.d + 32, self.h:self.h + 64,
                      self.w:self.w + 64]  # [16, 32, 32]

        return image_array, label_array

    def __len__(self):
        return len(self.image_list)


class SingleUnet3DData(Dataset):
    def __init__(self, h5_file, image_list, label_list):
        self.h_io = h5_file
        self.image_list = image_list
        self.label_list = label_list

        self.image_slice = []
        self.label_slice = []
        for image_name, label_name in zip(image_list, label_list):
            image_arr = np.transpose(np.array(self.h_io[image_name]), (2, 0, 1))
            label_arr = np.transpose(np.array(self.h_io[label_name]), (2, 0, 1))
            d, h, w = 160 // 32, 512 // 64, 512 // 64
            for i in range(d):
                for j in range(h):
                    for k in range(w):
                        self.image_slice.append(
                            image_arr[i * 32: i * 32 + 32, j * 64: j * 64 + 64, k * 64: k * 64 + 64])
                        self.label_slice.append(
                            label_arr[i * 32: i * 32 + 32, j * 64: j * 64 + 64, k * 64: k * 64 + 64])

    def __getitem__(self, index):
        image_array = torch.FloatTensor(self.image_slice[index]).unsqueeze(0)
        label_array = torch.LongTensor(self.label_slice[index])

        return image_array, label_array

    def __len__(self):
        return len(self.image_slice)


def getData(h5_file, num_img, num_for_each_img=4, single=False):
    if not single:
        datasets = []
        constrain_image, constrain_label = get_list(num_img * 5 // 8, h5_file)
        random_image, random_label = get_list(num_img - (num_img * 5 // 8), h5_file)
        for i in range(num_for_each_img):
            datasets.append(RandomUnet3DData(h5_file, random_image, random_label))
            datasets.append(ConstrainedUnet3DData(h5_file, constrain_image, constrain_label))
        return ConcatDataset(datasets)

    else:
        image, label = get_list(num_img, h5_file)
        return SingleUnet3DData(h5_file, image, label)


def load_data(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def evaluation_matrices(output, labels):
    IoU, Dice = 0, 0
    n_classes = output.shape[1]
    lab_pred = torch.argmax(output, dim=1) + 1
    pred_dict = Counter(np.array(lab_pred.flatten()))
    index = [(torch.ones_like(labels) * i == (labels + 1)).long() for i in range(1, n_classes + 1)]
    count_dict = [Counter(np.array(index[i] * lab_pred).flatten()) for i in range(n_classes)]
    TP = np.array([count_dict[i][i + 1] for i in range(n_classes)])
    FN = np.array([index[i].sum() - TP[i] for i in range(n_classes)])
    FP = np.array([pred_dict[i + 1] - TP[i] for i in range(n_classes)])

    IoU = TP / (TP + FP + FN + 1e-6)
    Dice = 2 * TP / (2 * TP + FP + FN + 1e-6)

    return IoU.mean(), Dice.mean()


def train(net, train_iter, test_iter, PCL, FDL, optimizer, path, num_loops=4, device="cuda"):
    net.to(device)
    for loop in range(num_loops):
        net.train()

        train_iou, train_dice, train_loss = [], [], []
        cnt = 1
        for (x, y) in tqdm(train_iter):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            feature_maps, output = net(x)
            loss = PCL(output, y) + 0.05 * FDL(feature_maps, y)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()

            if cnt % 16 == 0:
                state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, path)
            cnt += 1

            with torch.no_grad():
                tr_iou, tr_dice = evaluation_matrices(output, y)
                train_iou.append(tr_iou)
                train_dice.append(tr_dice)

        print(f"train_loss: {(sum(train_loss) / len(train_loss)):.5f},"
              f"train_IoU: {(sum(train_iou) / len(train_iou)):.5f}, "
              f"train_Dice: {(sum(train_dice) / len(train_dice)):.5f}")

        net.eval()
        test_iou, test_dice, test_loss = [], [], []
        with torch.no_grad():
            for (x, y) in tqdm(test_iter):
                x, y = x.to(device), y.to(device)
                feature_maps, output = net(x)
                l = PCL(output, y) + FDL(feature_maps, y)
                test_loss.append(l)
                te_iou, te_dice = evaluation_matrices(output, y)
                test_iou.append(te_iou)
                test_dice.append(te_dice)
        print(f"test_loss: {(sum(test_loss) / len(test_loss)):.5f},"
              f"test_IoU: {(sum(test_iou) / len(test_iou)):.5f}, "
              f"test_Dice: {(sum(test_dice) / len(test_dice)):.5f}")


def parse():
    parser = argparse.ArgumentParser("==== Vnet model ====")

    parser.add_argument("--h5_file", type=str, help="input directory of the dataset.")
    parser.add_argument("--dict_dir", type=str, help="temporary storage directory of the parameter dictionary.")
    parser.add_argument("--batch_size", type=int, help="batch size.")
    parser.add_argument("--num_images", type=int, help="the size of dataset.")
    parser.add_argument("--num_clips", type=int, help="the number of clips.")
    parser.add_argument("--single", type=bool, help="the test dataset is one single picture.")
    parser.add_argument("--num_epochs", type=int, help="training epochs.")
    parser.add_argument("--num_loops", type=int, help="training loops for each epoch.")
    parser.add_argument("--lr", type=float, help="learning rate.")
    parser.add_argument("--wd", type=float, help="weight decay rate.")

    return parser.parse_args()