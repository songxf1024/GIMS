import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as ts
from dgl.nn import SAGEConv
from .agc import build_optimize_graph_with_cosine_similarity, build_graph_from_keypoints_Delaunay


def MLP(channels: list, use_layernorm, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if use_layernorm:
                layers.append(LayerNorm(channels[i]))
            elif do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-2, keepdim=True)
        std = x.std(-2, keepdim=True)
        return torch.reshape(self.a_2, (1, -1, 1)) * ((x - mean) / (std + self.eps)) + torch.reshape(self.b_2, (1, -1, 1))

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs """
    def __init__(self, feature_dim, layers, score=False, use_layernorm=False):
        super().__init__()
        self.score = score
        self.encoder = MLP([3 if self.score else 2] + layers + [feature_dim], use_layernorm=use_layernorm)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)] if self.score else [kpts.transpose(1, 2),]
        return self.encoder(torch.cat(inputs, dim=1))

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, use_layernorm=False):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim], use_layernorm=use_layernorm)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, use_layernorm=False):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4, use_layernorm=use_layernorm)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, activation=F.relu, dropout=0.3):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, hidden_feats, 'mean'))
        for i in range(num_layers - 2): self.layers.append(SAGEConv(hidden_feats, hidden_feats, 'mean'))
        self.layers.append(SAGEConv(hidden_feats, out_feats, 'mean'))
        self.dropout = dropout
        self.activation = activation

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                # h = F.dropout(h, self.dropout, training=self.training)
        return h


class GMatcher(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights_path': None,
        'keypoint_encoder': [32, 64, 128, 256],
        'transformer_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'use_layernorm': False,
        'input_dim': 256,
        'num_heads': 4
    }
    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'],
            self.config['keypoint_encoder'],
            score=False,
            use_layernorm=self.config['use_layernorm'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'],
            self.config['transformer_layers'],
            use_layernorm=self.config['use_layernorm'])

        self.gnn_encoder = GraphSAGE(
            self.config['descriptor_dim'],
            int(self.config['descriptor_dim']/2),
            self.config['descriptor_dim'],
            3
        )
        if self.config["input_dim"] != self.config["descriptor_dim"]:
            self.input_proj = nn.Linear(self.config["input_dim"], self.config["descriptor_dim"], bias=True)
        else:
            self.input_proj = nn.Identity()
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'],
            self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        bin_score = torch.nn.Parameter(torch.tensor(1.0))
        self.register_parameter('bin_score', bin_score)
        if self.config['weights_path']:
            weights = torch.load(self.config['weights_path'], map_location="cpu", weights_only=False)
            if ('ema' in weights) and (weights['ema'] is not None):
                load_dict = weights['ema']
            elif 'model' in weights:
                load_dict = weights['model']
            else:
                load_dict = weights
            self.load_state_dict(load_dict)
            print('Loaded GMatcher model (\"{}\" weights)'.format(self.config['weights_path']))

    def forward(self, data, **kwargs):
        radius = data.get('radius', 25)
        percentile = data.get('percentile', 7)
        min_size = data.get('min_size', 8)
        if data.get('delaunay', False):
            # Delaunay method
            print("**Delaunay method**")
            t1 = time.time()
            graph0 = build_graph_from_keypoints_Delaunay(data['keypoints0'], data['descriptors0'], data['scores0'], device=data['device'])
            print('>> Graph Construction 1:', time.time() - t1)
            t1 = time.time()
            graph1 = build_graph_from_keypoints_Delaunay(data['keypoints1'], data['descriptors1'], data['scores1'], device=data['device'])
            print('>> Graph Construction 2:', time.time() - t1)
        else:
            # Dynamic threshold method
            t1 = time.time()
            graph0, kept_kpts0_indices = build_optimize_graph_with_cosine_similarity(data['keypoints0'], data['descriptors0'], data['scores0'],
                                                                                     radius=radius, percentile=percentile, min_size=min_size,
                                                                                     device=data['device'], image=data['image0'][0], show=False)
            print('>> Graph Construction 1:', time.time() - t1)
            t1 = time.time()
            graph1, kept_kpts1_indices = build_optimize_graph_with_cosine_similarity(data['keypoints1'], data['descriptors1'], data['scores1'],
                                                                                     radius=radius, percentile=percentile, min_size=min_size,
                                                                                     device=data['device'], image=data['image1'][0], show=False)
            print('>> Graph Construction 2:', time.time() - t1)
        data['keypoints0'] = torch.stack([g.ndata['point'] for g in graph0])
        data['descriptors0'] = torch.stack([g.ndata['feat'] for g in graph0]).permute(0, 2, 1)
        data['keypoints1'] = torch.stack([g.ndata['point'] for g in graph1])
        data['descriptors1'] = torch.stack([g.ndata['feat'] for g in graph1]).permute(0, 2, 1)
        data['scores0'] = torch.stack([g.ndata['score'] for g in graph0])
        data['scores1'] = torch.stack([g.ndata['score'] for g in graph1])
        data['kept_kpts0_indices'] = kept_kpts0_indices
        data['kept_kpts1_indices'] = kept_kpts1_indices
        data['graph0'], data['graph1'] = graph0, graph1

        if kwargs.get('mode', 'test') == "train": return self.forward_train(data)
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        # desc0, desc1 = data['descriptors0'], data['descriptors1']
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        t1 = time.time()
        desc0 = torch.stack([self.gnn_encoder(g, g.ndata['feat']) for g in graph0]).permute(0, 2, 1)
        desc1 = torch.stack([self.gnn_encoder(g, g.ndata['feat']) for g in graph1]).permute(0, 2, 1)
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])
        desc0, desc1 = self.gnn(desc0, desc1)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # sns.heatmap(scores.cpu().numpy()[0], ax=axs[0], cmap="viridis")
        # axs[0].set_title('Scores Before Optimal Transport')
        scores = log_optimal_transport( scores, self.bin_score, iters=self.config['sinkhorn_iterations'])
        # sns.heatmap(scores.cpu().numpy()[0], ax=axs[1], cmap="viridis")
        # axs[1].set_title('Scores After Optimal Transport')
        # plt.close()
        # plt.show()
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        print('>> Image Matching:', time.time() - t1)
        return {
            'keypoints0': data['keypoints0'],
            'keypoints1': data['keypoints1'],
            'descriptors0': data['descriptors0'],
            'descriptors1': data['descriptors1'],
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'mdesc0': mdesc0.permute(0, 2, 1).squeeze(),
            'mdesc1': mdesc1.permute(0, 2, 1).squeeze(),
        }

    def forward_train(self, data):
        """
        Optimize the loss function of the matching problem, adjust the score matrix through 
        the Sinkhorn algorithm to make it close to the optimal match, and then calculate the 
        loss of the positive and negative match based on the real match situation, and adjust 
        the loss contribution through the configured weights to train the model to perform 
        better on a given matching task.
        """
        graph0, graph1 = data['graph0'], data['graph1']
        batch_size = data['image0'].shape[0]
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        # desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        desc0 = torch.stack([self.gnn_encoder(g, g.ndata['feat']) for g in graph0]).permute(0, 2, 1)
        desc1 = torch.stack([self.gnn_encoder(g, g.ndata['feat']) for g in graph1]).permute(0, 2, 1)
        ## Transformer
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])
        desc0, desc1 = self.gnn(desc0, desc1)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # Calculate the inner product between two descriptors to generate a bnm-shaped tensor, where each element represents the similarity or distance score between a pair of descriptors
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        # Normalize the score by the square root of the descriptor dimension
        scores = scores / self.config['descriptor_dim']**.5
        # Sinkhorn algorithm optimizes the score matrix scores, using the optimal transmission method in logarithmic space
        scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])
        # gt_indexes is the index of the real match pair
        gt_indexes = data['matches']
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
        # Key points are deleted when composing, so the remaining key points need to be remapped.
        kept_kpts0_indices = data['kept_kpts0_indices']
        kept_kpts1_indices = data['kept_kpts1_indices']
        # Step 1: Construct a remap dictionary (batch-wise)
        remap_dict_0 = [
            {orig_idx: new_idx for new_idx, orig_idx in enumerate(kept_kpts0_indices[b])}
            for b in range(len(kept_kpts0_indices))
        ]
        remap_dict_1 = [
            {orig_idx: new_idx for new_idx, orig_idx in enumerate(kept_kpts1_indices[b])}
            for b in range(len(kept_kpts1_indices))
        ]
        # Step 2: Construct valid index row + index remapping
        gt_new_indexes = []
        for i in range(len(gt_indexes)):
            b, i0, i1 = gt_indexes[i].tolist()
            # Keep -1 first to indicate negative samples
            if i0 == -1 or i1 == -1:
                gt_new_indexes.append([b, -1, -1])
            else:
                # If the point is in the reserved index, the new index is mapped; otherwise it is marked as -1 (i.e. invalid)
                if (i0 in remap_dict_0[b]) and (i1 in remap_dict_1[b]):
                    new_i0 = remap_dict_0[b][i0]
                    new_i1 = remap_dict_1[b][i1]
                    gt_new_indexes.append([b, new_i0, new_i1])
                else:
                    gt_new_indexes.append([b, -1, -1])
        # Step 3: Convert to tensor
        gt_indexes = torch.tensor(gt_new_indexes, dtype=torch.long, device=gt_indexes.device)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
        # neg_flag marks which match pairs are negative (i.e., mismatched pairs), determine by checking whether the value in gt_indexes is -1
        neg_flag = (gt_indexes[:, 1] == -1) | (gt_indexes[:, 2] == -1)
        # Calculate the corresponding scores of each true match pair in the optimized score matrix
        loss_pre_components = scores[gt_indexes[:, 0], gt_indexes[:, 1], gt_indexes[:, 2]]
        # Clamping loss_pre_components, limiting their values ​​between -100 and 0 to avoid gradient explosion or numerical instability
        loss_pre_components = torch.clamp(loss_pre_components, min=-100, max=0.0)
        # Inverse the cropped scores to facilitate calculation of the loss (in the matching problem, higher scores indicate better matches, so the loss is a negative value of the score)
        loss_vector = -1 * loss_pre_components
        # According to neg_flag, pos_index and neg_index are indexes of positive and negative matches, respectively.
        neg_index, pos_index = gt_indexes[:, 0][neg_flag], gt_indexes[:, 0][~neg_flag]
        # Calculate the average of positive and negative match losses, respectively
        batched_pos_loss, batched_neg_loss = ts.scatter_mean(loss_vector[~neg_flag], pos_index, dim_size=batch_size), ts.scatter_mean(loss_vector[neg_flag], neg_index, dim_size=batch_size)
        # pos_loss and neg_loss are multiplied by the weights defined in the configuration, self.config['pos_loss_weight'] and self.config['neg_loss_weight'], respectively, 
        # to adjust the contribution of positive and negative matches to the total loss
        pos_loss, neg_loss = self.config['pos_loss_weight']*batched_pos_loss.mean(), self.config['neg_loss_weight']*batched_neg_loss.mean()
        # Total loss is the sum of positive matching loss pos_loss and negative matching loss neg_loss
        loss = pos_loss + neg_loss
        return loss, pos_loss, neg_loss

