import torch
import torch.nn as nn
from torch.nn import functional as F


def PCL(output, labels):
    CP_Loss = nn.CrossEntropyLoss()
    return CP_Loss(output, labels)


def FDL(maps, labels):
    FD_Loss = 0
    n_batches, n_filters = maps[0].shape[:2]
    index = [(torch.ones_like(labels) * i == labels).float() for i in range(n_filters)]  # change the pixel into 0 or 1

    for m in range(len(maps)):
        loss = []
        e_tilde = F.normalize(maps[m], dim=1)
        e_select = [(index[i].unsqueeze(1) * e_tilde).reshape(n_batches, n_filters, -1)
                    for i in range(n_filters)]  # (batches, filters, num_elements_in_3D_matirx)
        phi = [F.normalize(e_select[t].sum(dim=-1).unsqueeze(1), dim=-1)
               for t in range(1, n_filters)]  # (batches, 1, filters)

        for t in range(n_filters):
            if t:
                numerator = torch.exp(phi[t - 1].matmul(e_select[t])).squeeze()  # (batches, num_elements)
                denominator = torch.stack([torch.exp(phi[s].matmul(e_select[t])).squeeze()
                                           for s in range(n_filters - 1) if s != (t - 1)]).sum(0)  # (batches, num_elements)

            else:
                numerator = 1.0
                denominator = torch.stack([torch.exp(phi[s].matmul(e_select[t])).squeeze()
                                           for s in range(n_filters - 1)]).sum(0)

            if index[t].sum() == 0:
                d, h, w = labels.shape[-3:]
                l = -1.0 * ((torch.log(numerator / denominator)).sum() / (d * h * w))

            else:
                l = -1.0 * ((torch.log(numerator / denominator)).sum() / index[t].sum()) / n_batches  # (batches) -> 1 number

            loss.append(l)

        FD_Loss += sum(loss)

    return FD_Loss