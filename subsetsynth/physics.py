from __future__ import annotations

import torch
import torch.nn as nn


class DynamicBlochLayer(nn.Module):
    """Retains the v1.8.2 A/B acquisition prior only where parameters exist."""
    def __init__(self):
        super().__init__()
        self.register_buffer('TR_A', torch.tensor([634.0, 3002.0, 3500.0]))
        self.register_buffer('TE_A', torch.tensor([9.5, 109.0, 45.0]))
        self.register_buffer('TI_A', torch.tensor([0.0, 0.0, 160.0]))
        self.register_buffer('TR_B', torch.tensor([356.9, 2963.5, 3923.4]))
        self.register_buffer('TE_B', torch.tensor([8.8, 97.5, 56.0]))
        self.register_buffer('TI_B', torch.tensor([0.0, 0.0, 159.6]))

    @staticmethod
    def spin_echo(pd, t1, t2, TR, TE):
        return pd * (1.0 - torch.exp(-TR / t1)) * torch.exp(-TE / t2)

    @staticmethod
    def inversion_recovery(pd, t1, t2, TR, TE, TI):
        return pd * torch.abs(1.0 - 2.0 * torch.exp(-TI / t1) + torch.exp(-TR / t1)) * torch.exp(-TE / t2)

    def forward(self, phys_maps: torch.Tensor, cohort_id: torch.Tensor, jitter: bool = True):
        pd_map, t1_map, t2_map = phys_maps[:,0:1], phys_maps[:,1:2], phys_maps[:,2:3]
        B = phys_maps.shape[0]
        t1 = t1_map * 5000.0 + 10.0
        t2 = t2_map * 2500.0 + 10.0
        pd = pd_map * 2.0
        is_A = (cohort_id == 0).float().view(B,1,1,1)
        is_B = (cohort_id == 1).float().view(B,1,1,1)
        if self.training and jitter:
            j = torch.empty(B,1,1,1, device=phys_maps.device).uniform_(0.95,1.05)
        else:
            j = torch.ones(B,1,1,1, device=phys_maps.device)
        tr1 = (is_A*self.TR_A[0] + is_B*self.TR_B[0]) * j
        te1 = (is_A*self.TE_A[0] + is_B*self.TE_B[0]) * j
        tr2 = (is_A*self.TR_A[1] + is_B*self.TR_B[1]) * j
        te2 = (is_A*self.TE_A[1] + is_B*self.TE_B[1]) * j
        trf = (is_A*self.TR_A[2] + is_B*self.TR_B[2]) * j
        tef = (is_A*self.TE_A[2] + is_B*self.TE_B[2]) * j
        tif = (is_A*self.TI_A[2] + is_B*self.TI_B[2]) * j
        return torch.cat([
            self.spin_echo(pd,t1,t2,tr1,te1),
            self.spin_echo(pd,t1,t2,tr2,te2),
            self.inversion_recovery(pd,t1,t2,trf,tef,tif),
        ], dim=1)
