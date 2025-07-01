import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import math
import scipy.misc
import numpy as np
from detectron2.structures.triplets import Triplets

def extract_bbox(mask):
    horizontal_indicies = torch.where(torch.any(mask, dim=0))[0]
    vertical_indicies = torch.where(torch.any(mask, dim=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    box = torch.IntTensor([x1, y1, x2, y2])
    return box

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [num_instances, height, width]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (x1, y1, x2, y2)].
    """
    boxes = torch.zeros([mask.shape[0], 4])
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        # Bounding box.
        horizontal_indicies = torch.where(torch.any(m, axis=0))[0]
        vertical_indicies = torch.where(torch.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i][0]=x1
        boxes[i][1]=y1
        boxes[i][2]=x2
        boxes[i][3]=y2
    return boxes.int()

def union_box(box1,box2):
    min_x = torch.min(box1[0], box2[0])
    min_y = torch.min(box1[1], box2[1])
    max_x = torch.max(box1[2], box2[2])
    max_y = torch.max(box1[3], box2[3])
    new_box=torch.stack([min_x,min_y,max_x,max_y])
    return new_box

def union_boxes(boxes1,boxes2):
    if boxes1.shape[0]==0:
        return boxes1
    box_stack=torch.stack([boxes1,boxes2],dim=2) # x,4,2
    min_box = torch.min(box_stack,dim=2)[0] # x,4
    max_box = torch.max(box_stack,dim=2)[0] # x,4
    new_box=torch.stack([min_box[:,0],min_box[:,1],max_box[:,2],max_box[:,3]],dim=1)
    return new_box

def union_box_list(box_list):
    new_box = box_list[0]
    for i in range(1,box_list.shape[0]):
        new_box=union_box(new_box,box_list[i])
    return new_box

def union_box_matrix(box_matrix):
    new_boxes = box_matrix[:,0,:] # crowd,first instance,4
    for i in range(1, box_matrix.shape[1]):
        new_boxes = union_boxes(new_boxes, box_matrix[:,i,:])
    return new_boxes

def box_iou(box_pred, box_gt):
    area_pred=(box_pred[2]-box_pred[0])*(box_pred[3]-box_pred[1])
    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    area_sum = area_pred+area_gt

    left_line=max(box_pred[0],box_gt[0]) # x1
    top_line=max(box_pred[1],box_gt[1]) # y1
    right_line = min(box_pred[2], box_gt[2]) # x2
    bottom_line = min(box_pred[3], box_gt[3]) # y2

    if left_line>=right_line or top_line>=bottom_line:
        return 0
    else:
        intersect=(right_line-left_line)*(bottom_line-top_line)*1.0
        if intersect.item()==0:
            return 0
        # print(str(left_line)+" "+str(right_line)+" "+str(top_line)+" "+str(bottom_line))
        # print(str(intersect)+" "+str(area_sum-intersect))
        # print()
        u=area_sum-intersect
        u[u==0]=1.0
        intersect[u==0]=0.0
        return (intersect / u).item()

def compute_boxes_io(phrase_boxes, instance_boxes):
    phrase_ins=phrase_boxes.unsqueeze(1).repeat(1,instance_boxes.shape[0],1)
    phr_instances=instance_boxes.unsqueeze(0).repeat(phrase_boxes.shape[0],1,1)
    phrase_instance=torch.cat((phrase_ins, phr_instances),dim=2)

    instance_area=(instance_boxes[:,3]-instance_boxes[:,1])*(instance_boxes[:,2]-instance_boxes[:,0])
    instance_area=instance_area.unsqueeze(0).repeat(phrase_boxes.shape[0],1)

    intersection_x1 = torch.max(torch.stack((phrase_instance[:,:,0],phrase_instance[:,:,4]),dim=2),dim=2)[0]
    intersection_y1 = torch.max(torch.stack((phrase_instance[:,:,1],phrase_instance[:,:,5]),dim=2),dim=2)[0]
    intersection_x2 = torch.min(torch.stack((phrase_instance[:,:,2],phrase_instance[:,:,6]),dim=2),dim=2)[0]
    intersection_y2 = torch.min(torch.stack((phrase_instance[:,:,3],phrase_instance[:,:,7]),dim=2),dim=2)[0]

    intersection_x = intersection_x2 - intersection_x1
    intersection_y = intersection_y2 - intersection_y1
    intersection_x[intersection_x <= 0] = 0
    intersection_y[intersection_y <= 0] = 0
    intersection_area = intersection_x*intersection_y
    instance_area[instance_area == 0]=1.0
    intersection_area[instance_area==0]=0.0
    return intersection_area/instance_area

def boxes_iou(box_preds, box_gts):
    area_preds=(box_preds[:,2]-box_preds[:,0])*(box_preds[:,3]-box_preds[:,1])
    area_gts = (box_gts[:,2] - box_gts[:,0]) * (box_gts[:,3] - box_gts[:,1])
    area_sums = area_preds+area_gts

    left_lines = torch.max(box_preds[:,0],box_gts[:,0]) # x1
    top_lines = torch.max(box_preds[:,1],box_gts[:,1]) # y1
    right_lines = torch.min(box_preds[:,2], box_gts[:,2]) # x2
    bottom_lines = torch.min(box_preds[:,3], box_gts[:,3]) # y2

    intersect_hs = right_lines - left_lines
    intersect_hs = torch.where(intersect_hs < 0, torch.zeros_like(intersect_hs), intersect_hs)
    intersect_ws = bottom_lines - top_lines
    intersect_ws = torch.where(intersect_ws < 0, torch.zeros_like(intersect_ws), intersect_ws)
    intersect = intersect_hs * intersect_ws *1.0
    u=area_sums-intersect
    u[u == 0]=1.0
    intersect[u==0]=0.0
    return intersect / u

def mask_iou(mask_pred, mask_gt):
    intersection=mask_pred*mask_gt
    union=mask_pred+mask_gt
    if torch.sum(union>0).item()==0:
        return 0.0
    return (torch.sum(intersection>0)/torch.sum(union>0) * 1.0).item()

def compute_mask_io(phrase_mask_pred, phrase_mask_gt):
    intersection = phrase_mask_pred * phrase_mask_gt
    if torch.sum(phrase_mask_gt > 0).item()==0:
        return 0
    return (torch.sum(intersection > 0) / torch.sum(phrase_mask_gt > 0) * 1.0).item()

def compute_masks_io(phrase_mask_pred, phrase_mask_gt):
    pred_mul_gt_phrase_masks=phrase_mask_pred.unsqueeze(1).repeat(1,phrase_mask_gt.shape[0],1,1)
    mul_pred_gt_phrase_masks=phrase_mask_gt.unsqueeze(0).repeat(phrase_mask_pred.shape[0],1,1,1)
    # print(pred_mul_gt_phrase_masks.shape)
    # print(mul_pred_gt_phrase_masks.shape)
    intersection = pred_mul_gt_phrase_masks * mul_pred_gt_phrase_masks
    interaction_sum = torch.sum(torch.sum(intersection>0,dim=2),dim=2)*1.0
    gt_sum = torch.sum(torch.sum(mul_pred_gt_phrase_masks>0,dim=2),dim=2)
    interaction_sum[gt_sum==0]=0
    gt_sum[gt_sum==0]=1
    return interaction_sum/gt_sum

def calc_self_distance(X):
    rx=X.pow(2).sum(dim=-1).reshape((-1,1))
    dist=rx-2.0*X.matmul(X.t())+rx.t()
    dist[dist<0]=0
    return torch.sqrt(dist/X.shape[0])

def calc_pairwise_distance(X, Y):
    rx=X.pow(2).sum(dim=-1).reshape((-1,1))
    ry=Y.pow(2).sum(dim=-1).reshape((-1,1))
    dist=rx-2.0*X.matmul(Y.t())+ry.t()
    dist[dist<0]=0
    return torch.sqrt(dist)

class SelfGCNLayer(nn.Module): # self
    def __init__(self,source_channel,impact_channel):
        super(SelfGCNLayer, self).__init__()
        # self.source_fc=nn.Linear(source_channel,source_channel)
        self.impact_fc=nn.Linear(impact_channel,source_channel)

    def forward(self,source,impact,attention): # [n1,x1] [n2,x2] [n1,n2]
        result = F.relu(self.impact_fc(impact)) # [n2,x2]->[n2,x3]
        collect = attention @ result # [n1,n2]@[n2,x3]=[n1,x3]
        collect_avg = collect / (attention.sum(1).view(collect.shape[0], 1) + 1e-7)
        update=(source+collect_avg)/2
        return update

class OtherGCNLayer(nn.Module): # other
    def __init__(self,output_channel,input_channel):
        super(OtherGCNLayer, self).__init__()
        # self.source_fc=nn.Linear(source_channel,source_channel)
        self.impact_fc=nn.Linear(input_channel,output_channel)

    def forward(self,impact,attention): # [n2,x2] [n1,n2]
        result = F.relu(self.impact_fc(impact)) # [n2,x2]->[n2,x3]
        collect = attention @ result # [n1,n2]@[n2,x3]=[n1,x3]
        collect_avg = collect / (attention.sum(1).view(collect.shape[0], 1) + 1e-7)
        return collect_avg

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 10))
        # self.decoder = nn.Sequential(
        #     nn.Linear(10, 2000),
        #     nn.ReLU(True),
        #     nn.Linear(2000, 500),
        #     nn.ReLU(True),
        #     nn.Linear(500, 500),
        #     nn.ReLU(True),
        #     nn.Linear(500, 500),
        #     nn.ReLU(True),
        #     nn.Linear(500, 512))
        # self.model = nn.Sequential(self.encoder, self.decoder)
    def encode(self, x):
        return self.encoder(x)

    # def forward(self, x):
    #     x = self.model(x)
    #     return x

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
            self.n_clusters,
            self.hidden,
            dtype=torch.float
            ).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)
    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers)**2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t() #soft assignment using t-distribution
        return t_dist

class DEC(nn.Module):
    def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        self.cluster_centers = cluster_centers
        self.autoencoder = AutoEncoder()
        self.clusteringlayer = ClusteringLayer(self.n_clusters, self.hidden, self.cluster_centers, self.alpha)

    def target_distribution(self, q_):
        weight = (q_ ** 2) / torch.sum(q_, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def forward(self, x):
        x = self.autoencoder.encode(x)
        return self.clusteringlayer(x)

    def visualize(self, epoch,x):
        fig = plt.figure()
        ax = plt.subplot(111)
        x = self.autoencoder.encode(x).detach() 
        x = x.cpu().numpy()[:2000]
        x_embedded = TSNE(n_components=2).fit_transform(x)
        plt.scatter(x_embedded[:,0], x_embedded[:,1])
        fig.savefig('plots/mnist_{}.png'.format(epoch))
        plt.close(fig)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True,softmax=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.softmax = softmax

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        if len(h.shape)==2:
            Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        elif len(h.shape)==3:
            Wh = torch.bmm(h, self.W.unsqueeze(0).repeat(h.shape[0],1,1))  # h.shape: (B,N, in_features), Wh.shape: (B,N, out_features)
        else:
            print("not support h in GAT with shape:"+h.shape)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        if self.softmax:
            attention = F.softmax(attention, dim=1)
            similarity=attention
        else:
            similarity=F.sigmoid(attention)
            attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime),similarity
        else:
            return h_prime,similarity

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[-2]  # number of nodes
        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=-2)
        if len(Wh.shape)==2:
            Wh_repeated_alternating = Wh.repeat(N, 1)
        elif len(Wh.shape)==3:
            Wh_repeated_alternating = Wh.repeat(1, N, 1)
        else:
            print("shape of Wh is not supported:"+Wh.shape)
            exit()
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN
        # print(Wh_repeated_in_chunks.shape,Wh_repeated_alternating.shape)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)
        if len(Wh.shape)==2:
            return all_combinations_matrix.view(N, N, 2 * self.out_features)
        elif len(Wh.shape) == 3:
            return all_combinations_matrix.view(-1, N, N, 2 * self.out_features)
        else:
            print("shape of Wh is not supported:"+Wh.shape)
            exit()

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphAttentionLayerClass(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True,softmax=True):
        super(GraphAttentionLayerClass, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.softmax = softmax

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        if len(h.shape)==2:
            Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        elif len(h.shape)==3:
            Wh = torch.bmm(h, self.W.unsqueeze(0).repeat(h.shape[0],1,1))  # h.shape: (B,N, in_features), Wh.shape: (B,N, out_features)
        else:
            print("not support h in GAT with shape:"+h.shape)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        if self.softmax:
            attention = F.softmax(attention, dim=1)
            similarity=attention
        else:
            similarity=attention # F.sigmoid()
            attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime),similarity
        else:
            return h_prime,similarity

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout=256, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False,softmax=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        primes=[]
        for att in self.attentions:
            prime_out,attention_out = att(x,adj)
            primes.append(prime_out)
        x = torch.cat(primes, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        prime,attention=self.out_att(x, adj)
        x = F.elu(prime)

        return x, attention

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if len(input.shape)==2:
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support)
        elif len(input.shape)==3:
            support = torch.bmm(input, self.weight.unsqueeze(0).repeat(input.shape[0],1,1))
            output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0.0):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class CRF(nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags, batch_first= False):
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self):
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions, tags, mask = None, reduction = 'sum'):
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.type_as(emissions).sum()

    def decode(self, emissions, mask = None):
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(self,emissions,tags = None,mask = None):
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions, tags, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

# not used
class GAT_fc(nn.Module):
    def __init__(self, nfeat, nhid, nout=256, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GAT_fc, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False,softmax=False)
        self.similarity_fc1 = nn.Linear(nout * 2,nout)
        self.similarity_fc2 = nn.Linear(nout, nout)
        self.similarity_fc3 = nn.Linear(nout, 1)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        primes=[]
        # attentions=[]
        for att in self.attentions:
            prime_out,attention_out = att(x,adj)
            primes.append(prime_out)
            # attentions.append(attention_out)
        x = torch.cat(primes, dim=1)
        # attention=attentions[-1]
        x = F.dropout(x, self.dropout, training=self.training)
        prime,attention=self.out_att(x, adj)
        # print(prime)
        x = F.elu(prime)

        similarity_feature = torch.cat([x.unsqueeze(1).repeat(1, x.shape[0], 1),
                                        x.unsqueeze(0).repeat(x.shape[0], 1, 1)], dim=2)
        similarity_feature=F.elu(self.similarity_fc1(similarity_feature))
        similarity_feature = F.elu(self.similarity_fc2(similarity_feature))
        cat = self.similarity_fc3(similarity_feature).squeeze(2)

        return x, attention,F.sigmoid(cat) #F.log_softmax(x, dim=1)

# not used
class GATlast(nn.Module):
    def __init__(self, nfeat, nhid, nout=256, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GATlast, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * (nheads+1), nout, dropout=dropout, alpha=alpha, concat=False,softmax=False)
        self.similarity_fc = nn.Linear(nout * 2, 1)
        self.similarity_ac = nn.Sigmoid()

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        primes=[x]
        # attentions=[]
        for att in self.attentions:
            prime_out,attention_out = att(primes[-1],adj)
            primes.append(prime_out)
            # attentions.append(attention_out)
        x = torch.cat(primes, dim=1)
        # attention=attentions[-1]
        x = F.dropout(x, self.dropout, training=self.training)
        prime,attention=self.out_att(x, adj)
        x = F.elu(prime)
        similarity_feature = torch.cat([x.unsqueeze(1).repeat(1, x.shape[0], 1),
                                        x.unsqueeze(0).repeat(x.shape[0], 1, 1)], dim=2)
        cat = self.similarity_ac(self.similarity_fc(similarity_feature)).squeeze(2)
        return x, attention,cat #F.log_softmax(x, dim=1)

# not used
class GAT_su(nn.Module): # single update
    def __init__(self, nfeat, nhid, nout=256, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GAT_su, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, softmax=False)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.similarity_fc = nn.Linear(nout * 2, 1)
        self.similarity_fc = nn.Sigmoid()

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        prime_out,attention_out = self.attentions[0](x,adj)
        prime_out = F.elu(prime_out)
        similarity_feature = torch.cat([prime_out.unsqueeze(1).repeat(1, prime_out.shape[0], 1),
                                        prime_out.unsqueeze(0).repeat(prime_out.shape[0], 1, 1)], dim=2)
        cat = self.similarity_ac(self.similarity_fc(similarity_feature)).squeeze(2)
        return prime_out, attention_out, cat #F.log_softmax(x, dim=1)

# not used
class GAT_adjatt(nn.Module):
    def __init__(self, nfeat, nhid, nclass=1, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GAT_adjatt, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False,softmax=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        primes=[]
        attentions=[]
        for att in self.attentions:
            prime_out,attention_out = att(x,adj)
            primes.append(prime_out)
            attentions.append(attention_out)
        x = torch.cat(primes, dim=1)
        attention=attentions[-1]
        x = F.dropout(x, self.dropout, training=self.training)
        prime,attention=self.out_att(x, attention)
        x = F.elu(prime)
        return x, attention #F.log_softmax(x, dim=1)
# not used
class GAT_ml(nn.Module): # cat features
    def __init__(self, nfeat, nhid, nclass=1, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GAT_ml, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * (nheads+1), nclass, dropout=dropout, alpha=alpha, concat=False,softmax=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        primes=[x]
        attentions=[adj]
        for att in self.attentions:
            prime_out,attention_out = att(primes[-1],adj)
            primes.append(prime_out)
            attentions.append(attention_out)
        x = torch.cat(primes, dim=1)
        attention=attentions[-1]
        x = F.dropout(x, self.dropout, training=self.training)
        prime,attention=self.out_att(x, adj)
        x = F.elu(prime)
        return x, attention #F.log_softmax(x, dim=1)
# not used
class GAT_mllast(nn.Module): # latest feature
    def __init__(self, nfeat, nhid, nclass=1, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GAT_mllast, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=False,softmax=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        primes=[x]
        attentions=[adj]
        for att in self.attentions:
            prime_out,attention_out = att(primes[-1],adj)
            primes.append(prime_out)
            attentions.append(attention_out)
        x = primes[-1]
        attention=attentions[-1]
        x = F.dropout(x, self.dropout, training=self.training)
        prime,attention=self.out_att(x, adj)
        x = F.elu(prime)
        return x, attention #F.log_softmax(x, dim=1)
# not used
class GAT_mladjatt(nn.Module): # cat features
    def __init__(self, nfeat, nhid, nclass=1, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GAT_mladjatt, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * (nheads+1), nclass, dropout=dropout, alpha=alpha, concat=False,softmax=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        primes=[x]
        attentions=[adj]
        for att in self.attentions:
            prime_out,attention_out = att(primes[-1],adj)
            primes.append(prime_out)
            attentions.append(attention_out)
        x = torch.cat(primes, dim=1)
        attention=attentions[-1]
        x = F.dropout(x, self.dropout, training=self.training)
        prime,attention=self.out_att(x, attention)
        x = F.elu(prime)
        return x, attention #F.log_softmax(x, dim=1)
# not used
class GAT_mllastadjatt(nn.Module): # latest feature
    def __init__(self, nfeat, nhid, nclass=1, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GAT_mllastadjatt, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid, nclass, dropout=dropout, alpha=alpha, concat=False,softmax=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        primes=[x]
        attentions=[adj]
        for att in self.attentions:
            prime_out,attention_out = att(primes[-1],adj)
            primes.append(prime_out)
            attentions.append(attention_out)
        x = primes[-1]
        attention=attentions[-1]
        x = F.dropout(x, self.dropout, training=self.training)
        prime,attention=self.out_att(x, attention)
        x = F.elu(prime)
        return x, attention #F.log_softmax(x, dim=1)

# not used
class SimpleGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, softmax=True):
        super(SimpleGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.softmax = softmax

        self.trans = nn.Linear(in_features, out_features)
        self.attr = nn.Linear(out_features, 1)

    def forward(self, h, adj):
        trans_h=self.trans(h)
        a_input=trans_h.unsqueeze(1).repeat(1, trans_h.shape[0], 1)+trans_h.unsqueeze(0).repeat(trans_h.shape[0], 1, 1)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))
        attention = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
        if self.softmax:
            attention = F.softmax(attention, dim=1)
            similarity=attention
        else:
            similarity=F.sigmoid(attention)
            attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, trans_h)
        return h_prime,similarity
# not used
class GATSimple(nn.Module):
    def __init__(self, nfeat, nhid, nclass=1, dropout=0.6, alpha=0.2, nheads=1):
        """Dense version of GAT."""
        super(GATSimple, self).__init__()
        self.dropout = dropout

        self.attentions = [SimpleGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SimpleGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False,softmax=False)

    def forward(self, x, adj):
        primes=[]
        for att in self.attentions:
            prime_out,attention_out = att(x,adj)
            primes.append(prime_out)
        x = torch.cat(primes, dim=1)
        prime,attention=self.out_att(x, adj)
        return prime, attention