import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


def focal_loss(x, y):
    '''Focal loss.

    Args:
        x: (tensor) sized [N,D].
        y: (tensor) sized [N,].

    Return:
        (tensor) focal loss.
    '''
    alpha = 0.25
    gamma = 2

    t = one_hot_embedding(y.data.cpu(), self.num_classes)  # [N,21]
    t = Variable(t).cuda()  # [N,20]

    p = x.sigmoid()
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)



class MyLabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(MyLabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            # true_dist = torch.zeros_like(pred)
            # true_dist.fill_(self.smoothing / (self.cls - 1))
            # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

            true_dist = target.unsqueeze(1)
            second = true_dist * self.confidence + self.smoothing  # smooth
            first = 1 - second
            true_dist = torch.cat([first, second], 1)

        # print(true_dist[0])

        # print(target.shape)
        # print(target.max())
        # return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        return torch.mean(-true_dist * pred)


class MyBlurLabelSmoothingLoss(nn.Module):
    def __init__(self, classes, kernel_size, dim=-1):
        super(MyBlurLabelSmoothingLoss, self).__init__()
        # self.confidence = 1.0 - smoothing
        # self.smoothing = smoothing
        self.kernel_size = kernel_size
        self.cls = classes
        self.dim = dim

        self.layer = torch.nn.Conv2d(1, 1, kernel_size, 1, int(kernel_size / 2 - 0.5)).cuda()
        nn.init.constant_(self.layer.weight, 1)
        nn.init.constant_(self.layer.bias, 0)

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            # true_dist = torch.zeros_like(pred)
            # true_dist.fill_(self.smoothing / (self.cls - 1))
            # true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

            true_dist = target.unsqueeze(1).float()
            true_dist = self.layer(true_dist) / (self.kernel_size ** 2)

            # true_dist_np = true_dist.cpu().numpy()
            # true_dist_np_blur = cv2.blur(true_dist_np, (self.kernel_size, self.kernel_size))
            second = true_dist
            first = 1 - second
            true_dist = torch.cat([first, second], 1)

        # print(true_dist[0])

        # print(target.shape)
        # print(target.max())
        # return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        return torch.mean(-true_dist * pred)



class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # print(true_dist[0])

        # print(target.shape)
        # print(target.max())
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))