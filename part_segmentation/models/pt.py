import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from logger import get_missing_parameters_message, get_unexpected_parameters_message
from mamba_ssm import Mamba
from mamba_ssm.modules.mlp import GatedMLP
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from pointnet2_utils import PointNetFeaturePropagation
import rwkv6

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data[:,:,:3], number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)

class GLG(nn.Module):
    def __init__(self, dim, k_group_size, act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.group_size = k_group_size  # the first is the point itself

        self.affine_alpha_feat = nn.Parameter(
            torch.ones([1, 1, 1, dim * 2]))
        self.affine_beta_feat = nn.Parameter(
            torch.zeros([1, 1, 1, dim * 2]))

        self.mlp = nn.Sequential(norm_layer(dim * 2), nn.Linear(
            dim * 2, dim), act_layer())

    def forward(self, feat, idx, dist):
        B, _, C = feat.shape

        knn_x = feat.contiguous().view(-1, C)[idx, :]
        assert knn_x.shape[-1] == feat.shape[-1]
        knn_x = knn_x.view(
            B, feat.shape[1], self.group_size, feat.shape[-1]).contiguous()  # 1 128 8 384
        
        mean_x = feat.unsqueeze(dim=-2)
        std_x = torch.std(knn_x - mean_x)
        knn_x = (knn_x - mean_x) / (std_x + 1e-5)

        # Feature Expansion
        knn_x = torch.cat([knn_x, feat.reshape(
            B, feat.shape[1], 1, -1).repeat(1, 1, self.group_size, 1)], dim=-1)  # B G K 2C

        # Affine
        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat

        knn_x = knn_x * torch.exp(-torch.square(dist) / 2)
        knn_x = knn_x.max(2)[0]  # B G 2C
        knn_x = self.mlp(knn_x)

        return knn_x

class PointFinbaBlock(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=0.,
                 act_layer=nn.SiLU,
                 norm_layer=nn.LayerNorm,
                 k_group_size=8,
                 num_heads=6,
                 layer_id=None,
                 n_layer=12
                 ):
        super().__init__()

        self.layer_id = layer_id
        self.n_layer = n_layer
        self.k_group_size = k_group_size

        if self.layer_id == 0:
            self.ln0 = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.lfa = GLG(dim=dim,
                                k_group_size=self.k_group_size,
                                act_layer=act_layer,
                                norm_layer=norm_layer
                                )
        if layer_id % 2 == 1:
            self.mixer = Mamba(dim)
            self.mixer2 = GatedMLP(in_features=dim, activation=act_layer())
        else:
            self.mixer = rwkv6.RWKV_Tmix_x060(dim, num_heads, n_layer, layer_id)
            self.mixer2 = rwkv6.RWKV_CMix_x060(dim, n_layer, layer_id)

    def forward(self, x, idx, dist):
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.drop_path(self.lfa(self.norm1(x), idx, dist))
        x = x + self.drop_path(self.mixer(self.norm2(x)))
        x = x + self.drop_path(self.mixer2(self.norm3(x)))
        return x


class PointFinbaEncoder(nn.Module):
    def __init__(self, k_group_size=8, embed_dim=768, depth=4, drop_path_rate=0., num_heads=6):
        super().__init__()
        self.k_group_size = k_group_size
        self.num_heads = num_heads
        self.knn = KNN(k=k_group_size, transpose_mode=True)
        self.blocks = nn.ModuleList([
            PointFinbaBlock(
                dim=embed_dim,
                k_group_size=self.k_group_size,
                drop_path=drop_path_rate[i] if isinstance(
                    drop_path_rate, list) else drop_path_rate,
                num_heads=self.num_heads,
                layer_id=i,
                n_layer=depth
            )
            for i in range(depth)])

    def forward(self, center, x, pos):
        '''
        INPUT:
            x: patched point cloud and encoded, B G+1 C, 8 128+1=129 384
            pos: positional encoding, B G+1 C, 8 128+1=129 384
        OUTPUT:
            x: x after transformer block, keep dim, B G+1 C, 8 128+1=129 384

        NOTE: Remember adding positional encoding for every block, 'cause ptc is sensitive to position
        '''
        # get knn xyz and feature
        batch_size, num_points, _ = center.shape  # B N 3 : 1 128 3
        dist, idx = self.knn(center, center)  # B N K : get K idx for every center
        idx_base = torch.arange(
            0, batch_size, device=center.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        dist = dist.view(batch_size, num_points, self.k_group_size, 1).contiguous()

        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos, idx, dist)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class get_model(nn.Module):
    def __init__(self, cls_dim):
        super().__init__()

        self.trans_dim = 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.cls_dim = cls_dim
        self.num_heads = 6

        self.group_size = 32
        self.num_group = 128
        self.k_group_size = 4
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = 384
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = PointFinbaEncoder(
            embed_dim=self.trans_dim,
            k_group_size=self.k_group_size,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        self.propagation_0 = PointNetFeaturePropagation(in_channel=1152 + 3,
                                                        mlp=[self.trans_dim * 4, 1024])
        self.convs1 = nn.Conv1d(3392, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                        get_missing_parameters_message(incompatible.missing_keys)
                    )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                        get_unexpected_parameters_message(incompatible.unexpected_keys)

                    )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')
        else:
            print('Training from scratch')
            
    def forward(self, pts, cls_label):
        B, C, N = pts.shape
        pts = pts.transpose(-1, -2).contiguous()  # B N 3
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N
        pos = self.pos_embed(center)

        x = group_input_tokens
        # transformer
        feature_list = self.blocks(center, x, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1) #1152
        x_max = torch.max(x,2)[0]
        x_avg = torch.mean(x,2)
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1) #1152*2 + 64

        f_level_0 = self.propagation_0(pts.transpose(-1, -2), center.transpose(-1, -2), pts.transpose(-1, -2), x)

        x = torch.cat((f_level_0,x_global_feature), 1)
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss