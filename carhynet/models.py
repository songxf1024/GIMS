import math
import numpy as np
import torch
import torch.nn as nn
from .util import dist_th, cal_l2_distance_matrix, eps_sqrt, eps_l2_norm
import torch.nn.functional as F

# ---------------Common--------------------- #
def desc_l2norm(desc):
    '''descriptors with shape NxC or NxCxHxW'''
    # return (desc / torch.Tensor(desc).pow(2).sum(dim=1, keepdim=True).add(eps_l2_norm).pow(0.5)).numpy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    temp = desc
    if not torch.is_tensor(desc):
        temp = torch.Tensor(desc).to(device)
    if temp.ndim == 1:
        temp = temp.unsqueeze(0)
    temp = temp / temp.pow(2).sum(dim=1, keepdim=True).add(eps_l2_norm).pow(0.5)
    if torch.is_tensor(temp):
        return temp
    return temp.cpu().numpy()

class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_bias=True, is_scale=True, is_eps_leanable=False):
        """
        FRN layer as in the paper
        Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks'
        <https://arxiv.org/abs/1911.09737>
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable
        self.is_bias = is_bias
        self.is_scale = is_scale

        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        '''Additional information about customization of the printing module'''
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        # Compute the mean norm of activations per channel.
        # Calculate the average of activations of each channel 
        # The operation of FRN is in the (H, W) dimension, that is, each channel of each sample is normalized separately. 
        # Here x is a vector of N-dimensional (HxW), so FRN does not have the problem of BN layer dependence on batch. 
        # The normalization method of the BN layer is to subtract the mean and divide by the standard deviation, while the FRN is different. There is no subtracting the mean operation here. 
        # v^2 in the formula is the average of the quadratic norm of x. 
        # This normalization method is similar to that of BN that can be used to eliminate the scale problems caused by intermediate operations (convolution and nonlinear activation), which helps model training.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        # The epsilon in the formula is a very small normal quantity to prevent dividing 0. 
        # FRN is normalized in the two dimensions of H and W. Generally speaking, the network's feature map size N=HxW is larger, 
        # However, sometimes 1x1 may occur, such as InceptionV3 and VGG networks, and epsilon is more critical at this time. 
        # When the epsilon value is small, normalization is equivalent to a sign function. At this time, the gradient is almost 0, which seriously affects model training; 
        # When the value is large, the curve becomes smoother, and the gradient at this time is conducive to model learning. 
        # For this case, the paper recommends using a learnable epsilon. 
        # For models that do not contain 1x1 features, a constant value of 1e-6 is used in the paper.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        # After normalization, scaling and translation transformation are also required. Here, gamma and beta are also learnable parameters (parameter size is C)
        if self.is_scale:
            x = self.weight * x
        if self.is_bias:
            x = x + self.bias
        return x

class TLU(nn.Module):
    def __init__(self, num_features):
        """
        TLU layer as in the paper
        FRN lacks a de-meaning operation, which may cause the normalization to shift arbitrarily by 0. 
        If the FRN is followed by the ReLU activation layer, many 0 values ​​may be generated, which is detrimental to model training and performance. 
        To solve this problem, the thresholded ReLU, namely, TLU, is adopted after FRN. Here tau is a learnable parameter. 
        The paper found that the use of TLU after FRN is crucial to improve performance.
        Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks'
        <https://arxiv.org/abs/1911.09737>
        """
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.zeros_(self.tau)
        nn.init.constant_(self.tau, -1)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        #self.relu = nn.SiLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None: min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v: new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class SandGlass(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, identity_tensor_multiplier=1.0, norm_layer=None, keep_3x3=False):
        """
        x: Input tensor oup: Output channel number 
        stride: Step size expand_ratio: Expansion coefficient 
        identity_tensor_multiplier: Floating point number with intervals of 0-1, used for partial channel residual connections, default is 1, that is, the original residual connection 
        norm_layer: If False, no BN is used, and if True, a layer of BN is passed
        """
        super(SandGlass, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_identity = False if identity_tensor_multiplier == 1.0 else True
        self.identity_tensor_channels = int(round(inp * identity_tensor_multiplier))
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        # dw
        if expand_ratio == 2 or inp == oup or keep_3x3:
            layers.append(ConvBNReLU(inp, inp, kernel_size=3, stride=1, groups=inp, norm_layer=norm_layer))
        layers.append(CoordAtt(inp, inp))
        if expand_ratio != 1:
            # pw-linear
            layers.extend([
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                norm_layer(hidden_dim),
            ])
        layers.extend([
            # pw
            ConvBNReLU(hidden_dim, oup, kernel_size=1, stride=1, groups=1, norm_layer=norm_layer),
        ])
        if expand_ratio == 2 or inp == oup or keep_3x3 or stride == 2:
            layers.extend([
                # dw-linear
                nn.Conv2d(oup, oup, kernel_size=3, stride=stride, groups=oup, padding=1, bias=False),
                norm_layer(oup),
            ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.use_identity:
                identity_tensor = x[:, :self.identity_tensor_channels, :, :] + out[:, :self.identity_tensor_channels, :, :]
                out = torch.cat([identity_tensor, out[:, self.identity_tensor_channels:, :, :]], dim=1)
                # out[:,:self.identity_tensor_channels,:,:] += x[:,:self.identity_tensor_channels,:,:]
            else:
                out = x + out
            return out
        else:
            return out

# ------------------------------------------- #

# ---------------BaseNet--------------------- #
class BaseOriSingle8Net(nn.Module):
    def input_norm(self, x):
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        # WARNING: We need the .detach() input, otherwise the gradient generated by patches extractor 
        # with F.grid_sample is very noisy, making detector training completely unstable.
        return (x - mean.detach()) / (std.detach() + 1e-7)

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            # print(m.weights.sum())
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

    def forward(self, x, mode='eval'):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]:
            x = layer(x)
        desc_raw = self.layer8(x).squeeze()
        desc = desc_l2norm(desc_raw)

        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc

class BaseOriMultiNet(nn.Module):
    def forward(self, x, mode='eval'):
        for layer in [self.layer1, self.layer2, self.layer2, self.layer3, self.layer4, self.layer4, self.layer5, self.layer6]:
            x = layer(x)
        desc_raw = self.layer7(x).squeeze()
        desc = desc_l2norm(desc_raw)

        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc

class BaseResNet(nn.Module):
    def input_norm(self, x):
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        # WARNING: We need the .detach() input, otherwise the gradient generated by patches extractor 
        # with F.grid_sample is very noisy, making detector training completely unstable.
        return (x - mean.detach()) / (std.detach() + 1e-7)

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            # print(m.weights.sum())
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

    def forward(self, x, mode='eval'):
        x1 = self.block1(self.input_norm(x))
        x2 = self.block2(x1)
        g1 = self.global_pool2(x2)
        g2 = self.global_pool1(x1)
        desc_raw = g1 + g2
        desc_raw = desc_raw.squeeze()
        desc = F.normalize(desc_raw, p=2, dim=1)
        # desc = desc_l2norm(desc_raw)
        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc
# ------------------------------------------- #

# ---------------Model Def------------------- #
class CAR_HyNet(nn.Module):
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.2):
        super(CAR_HyNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.layer1 = nn.Sequential(
            FRN(3, is_bias=is_bias_FRN),
            TLU(3),
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            CoordAtt(32, 32),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            CoordAtt(32, 32),
            TLU(32),
        )
        self.layer2_5 = SandGlass(32, 32, 1, 6)

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer4_5 = SandGlass(64, 64, 1, 6)

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )


    def input_norm(self, x):
        std, mean = torch.std_mean(x, dim=(2, 3), keepdim=True)
        # WARNING: We need the .detach() input, otherwise the gradient generated by patches extractor 
        # with F.grid_sample is very noisy, making detector training completely unstable.
        return (x - mean.detach()) / (std.detach() + 1e-7)

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=0.6)
            try:
                nn.init.constant_(m.bias.data, 0.01)
            except:
                pass

    def forward(self, x, mode='eval'):
        #for layer in [self.layer1, self.layer2, self.layer2, self.layer3, self.layer4, self.layer4, self.layer5, self.layer6]:
        #    x = layer(x)
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer2_5(x1)
        x3 = x1 + x2
        x = self.layer3(x3)
        x1 = self.layer4(x)
        x2 = self.layer4_5(x1)
        x3 = x1 + x2
        x = self.layer5(x3)
        x = self.layer6(x)

        desc_raw = self.layer7(x).squeeze()
        desc = desc_l2norm(desc_raw)

        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc

class HyNet(nn.Module):
    """HyNet model definition"""
    def __init__(self, is_bias=True, is_bias_FRN=True, dim_desc=128, drop_rate=0.2):
        super(HyNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        self.layer1 = nn.Sequential(
            FRN(1, is_bias=is_bias_FRN),
            TLU(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            FRN(32, is_bias=is_bias_FRN),
            TLU(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            FRN(64, is_bias=is_bias_FRN),
            TLU(64),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            FRN(128, is_bias=is_bias_FRN),
            TLU(128),
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )

    def forward(self, x, mode='eval'):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6]:
            x = layer(x)
        desc_raw = self.layer7(x).squeeze()
        desc = desc_l2norm(desc_raw)
        if mode == 'train':
            return desc, desc_raw
        elif mode == 'eval':
            return desc

class L2Net(nn.Module):
    """L2Net model definition"""
    def __init__(self, is_bias=False, is_affine=False, dim_desc=128, drop_rate=0.3):
        super(L2Net, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            nn.InstanceNorm2d(1, affine=is_affine),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(32, affine=is_affine),
            activation,
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(32, affine=is_affine),
            activation,
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            norm_layer(64, affine=is_affine),
            activation,
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(64, affine=is_affine),
            activation,
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            norm_layer(128, affine=is_affine),
            activation,
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(128, affine=is_affine),
            activation,
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False)
        )

        return

    def forward(self, x):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]:
            x = layer(x)
        return desc_l2norm(x.squeeze())

class Loss_HyNet():
    def __init__(self, device, num_pt_per_batch, dim_desc, margin, alpha, is_sosr, knn_sos=8):
        self.device = device
        self.margin = margin
        self.alpha = alpha
        self.is_sosr = is_sosr
        self.num_pt_per_batch = num_pt_per_batch
        self.dim_desc = dim_desc
        self.knn_sos = knn_sos
        self.index_desc = torch.LongTensor(range(0, num_pt_per_batch))
        self.index_dim = torch.LongTensor(range(0, dim_desc))
        diagnal = torch.eye(num_pt_per_batch)
        self.mask_pos_pair = diagnal.eq(1).float().to(self.device)
        self.mask_neg_pair = diagnal.eq(0).float().to(self.device)

    def sort_distance(self):
        L = self.L.clone().detach()
        L = L + 2 * self.mask_pos_pair
        L = L + 2 * L.le(dist_th).float()
        R = self.R.clone().detach()
        R = R + 2 * self.mask_pos_pair
        R = R + 2 * R.le(dist_th).float()
        LR = self.LR.clone().detach()
        LR = LR + 2 * self.mask_pos_pair
        LR = LR + 2 * LR.le(dist_th).float()
        self.indice_L = torch.argsort(L, dim=1)
        self.indice_R = torch.argsort(R, dim=0)
        self.indice_LR = torch.argsort(LR, dim=1)
        self.indice_RL = torch.argsort(LR, dim=0)
        return

    def triplet_loss_hybrid(self):
        L = self.L
        R = self.R
        LR = self.LR
        indice_L = self.indice_L[:, 0]
        indice_R = self.indice_R[0, :]
        indice_LR = self.indice_LR[:, 0]
        indice_RL = self.indice_RL[0, :]
        index_desc = self.index_desc
        dist_pos = LR[self.mask_pos_pair.bool()]
        dist_neg_LL = L[index_desc, indice_L]
        dist_neg_RR = R[indice_R, index_desc]
        dist_neg_LR = LR[index_desc, indice_LR]
        dist_neg_RL = LR[indice_RL, index_desc]
        dist_neg = torch.cat((dist_neg_LL.unsqueeze(0),
                              dist_neg_RR.unsqueeze(0),
                              dist_neg_LR.unsqueeze(0),
                              dist_neg_RL.unsqueeze(0)), dim=0)
        dist_neg_hard, index_neg_hard = torch.sort(dist_neg, dim=0)
        dist_neg_hard = dist_neg_hard[0, :]
        # scipy.io.savemat('dist.mat', dict(dist_pos=dist_pos.cpu().detach().numpy(), dist_neg=dist_neg_hard.cpu().detach().numpy()))
        # Limited data range
        loss_triplet = torch.clamp(self.margin + (dist_pos + dist_pos.pow(2)/2*self.alpha) - (dist_neg_hard + dist_neg_hard.pow(2)/2*self.alpha), min=0.0)
        self.num_triplet_display = loss_triplet.gt(0).sum()
        self.loss = self.loss + loss_triplet.sum()
        self.dist_pos_display = dist_pos.detach().mean()
        self.dist_neg_display = dist_neg_hard.detach().mean()
        return

    def norm_loss_pos(self):
        diff_norm = self.norm_L - self.norm_R
        self.loss += diff_norm.pow(2).sum().mul(0.1)

    def sos_loss(self):
        L = self.L
        R = self.R
        knn = self.knn_sos
        indice_L = self.indice_L[:, 0:knn]
        indice_R = self.indice_R[0:knn, :]
        indice_LR = self.indice_LR[:, 0:knn]
        indice_RL = self.indice_RL[0:knn, :]
        index_desc = self.index_desc
        num_pt_per_batch = self.num_pt_per_batch
        index_row = index_desc.unsqueeze(1).expand(-1, knn)
        index_col = index_desc.unsqueeze(0).expand(knn, -1)
        A_L = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_R = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_LR = torch.zeros(num_pt_per_batch, num_pt_per_batch).to(self.device)
        A_L[index_row, indice_L] = 1
        A_R[indice_R, index_col] = 1
        A_LR[index_row, indice_LR] = 1
        A_LR[indice_RL, index_col] = 1
        A_L = A_L + A_L.t()
        A_L = A_L.gt(0).float()
        A_R = A_R + A_R.t()
        A_R = A_R.gt(0).float()
        A_LR = A_LR + A_LR.t()
        A_LR = A_LR.gt(0).float()
        A = A_L + A_R + A_LR
        A = A.gt(0).float() * self.mask_neg_pair
        sturcture_dif = (L - R) * A
        self.loss = self.loss + sturcture_dif.pow(2).sum(dim=1).add(eps_sqrt).sqrt().sum()
        return

    def compute(self, desc_L, desc_R, desc_raw_L, desc_raw_R):
        # num_pt_current_batch = desc_L.size(0)
        # # Dynamically create masks
        # diagonal = torch.eye(num_pt_current_batch, device=self.device)
        # self.mask_pos_pair = diagonal.eq(1).float()
        # self.mask_neg_pair = diagonal.eq(0).float()
        # self.index_desc = torch.LongTensor(range(0, num_pt_current_batch))
        self.desc_L = desc_L
        self.desc_R = desc_R
        self.desc_raw_L = desc_raw_L
        self.desc_raw_R = desc_raw_R
        self.norm_L = self.desc_raw_L.pow(2).sum(1).add(eps_sqrt).sqrt()
        self.norm_R = self.desc_raw_R.pow(2).sum(1).add(eps_sqrt).sqrt()
        self.L = cal_l2_distance_matrix(desc_L, desc_L)
        self.R = cal_l2_distance_matrix(desc_R, desc_R)
        self.LR = cal_l2_distance_matrix(desc_L, desc_R)
        self.loss = torch.Tensor([0]).to(self.device)
        self.sort_distance()
        self.triplet_loss_hybrid()
        self.norm_loss_pos()
        if self.is_sosr: self.sos_loss()
        return self.loss, self.dist_pos_display, self.dist_neg_display
# ------------------------------------------- #

class HyNetnetFeature2D:
    def __init__(self, do_cuda=True, cuda="cuda:0"):
        self.G_dim = 128
        self.do_cuda = do_cuda and torch.cuda.is_available()
        self.device = torch.device(cuda if self.do_cuda else "cpu")
        self.model_weights_path = './weights/car_hynet.pth'
        self.model = CAR_HyNet().to(self.device)
        self.mag_factor = 1.0
        self.batch_size = 512
        self.process_all = True
        self.checkpoint = torch.load(self.model_weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(self.checkpoint)
        if self.do_cuda: self.model.to(self.device)
        self.model.eval()
        print('==> CAR-HyNet successfully loaded pre-trained network.')

    def compute_des_batches(self, patches, color):
        descriptors = np.zeros((len(patches), self.G_dim), dtype=np.float32)
        for i in range(0, len(patches), self.batch_size):
            data_a = patches[i: i + self.batch_size, :, :].astype(np.float32)
            if color: data_a = torch.from_numpy(data_a).permute(0, 3, 1, 2)
            else: data_a = torch.from_numpy(data_a).unsqueeze(1)
            if self.do_cuda: data_a = data_a.to(self.device)
            with torch.no_grad():
                out_a = self.model(data_a)
                descriptors[i: i + self.batch_size] = out_a.cpu().detach().numpy().reshape(-1, self.G_dim)
        return descriptors

    def compute_sift(self, patches, kps, color=True):
        if len(kps) == 0: return kps, []
        decs = self.compute_des_batches(patches, color).astype(np.float32)
        return kps, decs

