import torch


def FloatTensor(data, cuda):
    if cuda:
        return torch.cuda.FloatTensor(data)
    else:
        return torch.FloatTensor(data)


def LongTensor(data, cuda):
    if cuda:
        return torch.cuda.LongTensor(data)
    else:
        return torch.LongTensor(data)