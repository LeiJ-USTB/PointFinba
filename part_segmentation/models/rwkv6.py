
# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

from mmengine.model import BaseModule

from timm.models.layers import DropPath

logger = logging.getLogger(__name__)

T_MAX = 256
HEAD_SIZE = 64

from torch.utils.cpp_extension import load
wkv6_cuda = load(name="wkv6",
                 sources=["models/rwkv_cuda/wkv6_op.cpp", 
                          "models/rwkv_cuda/wkv6_cuda.cu"],
                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math",
                 "-O3", "-Xptxas -O3", 
                 "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", 
                 f"-D_T_={T_MAX}"])

class WKV_6(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, dim=0).view(H, C//H)
            return (None, None, None, None, gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)
    
class RWKV_Tmix_x060(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, layer_id, bias=False):
        super().__init__()
        self.layer_id = layer_id
        self.dim_att = n_embd

        self.n_head = n_head
        self.head_size = self.dim_att // self.n_head
        assert self.dim_att % self.n_head == 0
        
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, n_embd).uniform_(-0.01, 0.01))
            
            # fancy time_decay
            decay_speed = torch.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -6 + 5 * (n / (self.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, self.dim_att).uniform_(-0.01, 0.01))
            
            tmp = torch.zeros(self.dim_att)
            for n in range(self.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (self.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(n_embd, self.dim_att, bias=bias)
        self.key = nn.Linear(n_embd, self.dim_att, bias=bias)

        self.value = nn.Linear(n_embd, self.dim_att, bias=bias)
        self.output = nn.Linear(self.dim_att, n_embd, bias=bias)
        self.gate = nn.Linear(n_embd, self.dim_att, bias=bias)
        self.ln_x = nn.GroupNorm(self.n_head, self.dim_att, eps=1e-5)

    def forward(self, x):
        B, T, C = x.shape
        xx = self.time_shift(x) - x
        
        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)

        x = x.view(B * T, C)        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)

        return x

class RWKV_CMix_x060(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id,
                 hidden_rate=4, bias=False):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=bias)
        self.receptance = nn.Linear(n_embd, n_embd, bias=bias)
        self.value = nn.Linear(hidden_sz, n_embd, bias=bias)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(xr)) * kv

        return x