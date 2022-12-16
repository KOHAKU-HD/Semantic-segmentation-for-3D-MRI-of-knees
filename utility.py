import random


def get_list(num_elements, h5_file):
    image_list = ["patient/" + name + "/series/0/img" for name in h5_file["patient"]]
    label_list = ["patient/" + name + "/series/0/seg" for name in h5_file["patient"]]
    seq = list(range(151))
    random.shuffle(seq)
    image_list = [image_list[i] for i in seq[:num_elements]]
    label_list = [label_list[i] for i in seq[:num_elements]]

    return image_list, label_list