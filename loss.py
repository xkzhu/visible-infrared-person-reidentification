import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def mean_weights( mask):
    sum_col = torch.sum(mask, dim=1, keepdim=True)[0]
    W = mask / sum_col
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

class TripletLoss_mean(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_mean, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = mean_weights(is_pos)
        weights_an = mean_weights(is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

class TripletLoss_Convexhull(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_Convexhull, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N, dim = dist_mat.size()
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        api = []
        ani = []

        for i in range(N):
            api.append(torch.sum(torch.matmul(torch.diag(weights_ap[i, :]), inputs),dim=0,keepdim=True))
            ani.append(torch.sum(torch.matmul(torch.diag(weights_an[i, :]), inputs),dim=0,keepdim=True))
        input_pos = torch.cat(api, dim=0)
        input_neg = torch.cat(ani, dim=0)
        furthest_positive = dist_pairwise(inputs,input_pos)
        closest_negative = dist_pairwise(inputs,input_neg)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def dist_pairwise(emb1, emb2):
    '''
    compute the eucilidean distance  between the corresponding samples of embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1 - emb2, 2).sum(dim=1, keepdim=True)
    dist = emb1_pow.sqrt()
    return dist


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis=1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis=1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx


# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label_id):
        num = label_id.size(0)
        label = torch.cat([torch.zeros(1, num), torch.ones(1, num)], dim=1).cuda()
        output1_ap = []
        output2 = []
        output2_an = []
        pre_class = -1
        for i in range(num):
            current_class = label_id[i]
            class_i = output[label_id == current_class, :]
            output1_ap.append(torch.mean(class_i, dim=0, keepdim=True))
            if (current_class != pre_class):
                output2.append(torch.mean(class_i, dim=0, keepdim=True))
                class_j = output[label_id != label_id[i], :]
                output2_an.append(torch.mean(class_j, dim=0, keepdim=True))
        output2 = torch.cat(output2, dim=0)
        output1_pos = torch.cat(output1_ap, dim=0)
        output2_neg = torch.cat(output2_an, dim=0)

        output = torch.cat([output, output2], dim=0)
        output2 = torch.cat([output1_pos, output2_neg], dim=0)
        euclidean_distance = F.pairwise_distance(output, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class ScatterLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=1.0):
        super(ScatterLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label_id):
        num = label_id.size(0)
        output1_ap = []
        output2 = []
        output2_an = []
        pre_class = -1
        for i in range(num):
            current_class = label_id[i]
            class_i = output[label_id == current_class, :]
            if (current_class != pre_class):
                output2.append(torch.mean(class_i, dim=0, keepdim=True))
                class_j = output[label_id != label_id[i], :]
                output2_an.append(torch.mean(class_j, dim=0, keepdim=True))
        output2 = torch.cat(output2, dim=0)
        output2_neg = torch.cat(output2_an, dim=0)

        euclidean_distance = F.pairwise_distance(output2, output2_neg, keepdim=True)
        loss_contrastive = torch.mean(torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class NearLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(NearLoss, self).__init__()
        self.margin = margin

    def forward(self, output, label_id):
        num = label_id.size(0)
        output_pos = []
        for i in range(num):
            class_i = output[label_id == label_id[i], :]
            output_pos.append(torch.mean(class_i, dim=0, keepdim=True))
        output2 = torch.cat(output_pos, dim=0)
        euclidean_distance = F.pairwise_distance(output, output2, keepdim=True)
        loss_near = torch.mean(torch.pow(euclidean_distance, 2))
        return loss_near
