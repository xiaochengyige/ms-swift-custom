# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, Sequence, Union

import einops
import torch
from torch import nn


def so3_to_yaw_torch(rot_mat: torch.Tensor) -> torch.Tensor:
    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return torch.atan2(cos_th_sin_phi, cos_th_cos_phi)


def round_2pi_torch(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    delta = torch.diff(phi, dim=-1)
    delta = round_2pi_torch(delta)
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(delta, dim=-1)], dim=-1)


def first_order_D(
    num_steps: int,
    lead_shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    matrix = torch.zeros(*lead_shape, num_steps - 1, num_steps, dtype=dtype, device=device)
    rows = torch.arange(num_steps - 1, device=device)
    matrix[..., rows, rows] = -1.0
    matrix[..., rows, rows + 1] = 1.0
    return matrix


def second_order_D(
    num_steps: int,
    lead_shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    matrix = torch.zeros(*lead_shape, max(num_steps - 2, 0), num_steps, dtype=dtype, device=device)
    rows = torch.arange(max(num_steps - 2, 0), device=device)
    matrix[..., rows, rows] = -1.0
    matrix[..., rows, rows + 1] = 2.0
    matrix[..., rows, rows + 2] = -1.0
    return matrix


def third_order_D(
    num_steps: int,
    lead_shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    matrix = torch.zeros(*lead_shape, max(num_steps - 3, 0), num_steps, dtype=dtype, device=device)
    rows = torch.arange(max(num_steps - 3, 0), device=device)
    matrix[..., rows, rows] = -1.0
    matrix[..., rows, rows + 1] = 3.0
    matrix[..., rows, rows + 2] = -3.0
    matrix[..., rows, rows + 3] = 1.0
    return matrix


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def construct_dtd(
    num_steps: int,
    lead_shape: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
    w_smooth1: Optional[Union[float, torch.Tensor]] = None,
    w_smooth2: Optional[Union[float, torch.Tensor]] = None,
    w_smooth3: Optional[Union[float, torch.Tensor]] = None,
    lam: float = 1e-3,
    dt: float = 1.0,
) -> torch.Tensor:
    dtd = torch.zeros(*lead_shape, num_steps, num_steps, dtype=dtype, device=device)
    for order, weights in ((1, w_smooth1), (2, w_smooth2), (3, w_smooth3)):
        if weights is None:
            continue
        if isinstance(weights, float):
            width = max(num_steps - order, 0)
            weights = torch.full((*lead_shape, width), weights, dtype=dtype, device=device)
        if order == 1:
            D = first_order_D(num_steps, lead_shape, device=device, dtype=dtype)
            scale = lam / dt**2
        elif order == 2:
            D = second_order_D(num_steps, lead_shape, device=device, dtype=dtype)
            scale = lam / dt**4
        else:
            D = third_order_D(num_steps, lead_shape, device=device, dtype=dtype)
            scale = lam / dt**6
        dtd += scale * einops.einsum(
            D * weights.unsqueeze(-1),
            D,
            "... i j, ... i k -> ... j k",
        )
    return dtd


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def solve_single_constraint(
    x_init: torch.Tensor,
    x_target: torch.Tensor,
    w_data: Optional[torch.Tensor] = None,
    w_smooth1: Optional[Union[float, torch.Tensor]] = None,
    w_smooth2: Optional[Union[float, torch.Tensor]] = None,
    w_smooth3: Optional[Union[float, torch.Tensor]] = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    device, dtype = x_target.device, x_target.dtype
    *lead_shape, num_steps = x_target.shape
    if w_data is None:
        w_data = torch.ones_like(x_target)
    x_init = torch.as_tensor(x_init, dtype=dtype, device=device)

    A = torch.eye(num_steps, dtype=dtype, device=device).expand(*lead_shape, num_steps, num_steps)
    weighted_A = A * w_data.unsqueeze(-1)
    ata = einops.einsum(weighted_A, A, "... i j, ... i k -> ... j k")
    rhs = einops.einsum(weighted_A, x_target, "... i j, ... i -> ... j")

    dtd = construct_dtd(
        num_steps + 1,
        lead_shape,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )
    rhs -= dtd[..., 1:, 0] * x_init.unsqueeze(-1)

    ridge_term = ridge * torch.eye(num_steps, dtype=dtype, device=device).expand(
        *lead_shape, num_steps, num_steps
    )
    lhs = ata + dtd[..., 1:, 1:] + ridge_term
    L = torch.linalg.cholesky(lhs)
    solution = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    return torch.cat([x_init.unsqueeze(-1), solution], dim=-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def solve_xs_eq_y(
    slope: torch.Tensor,
    target: torch.Tensor,
    w_data: Optional[torch.Tensor] = None,
    w_smooth1: Optional[Union[float, torch.Tensor]] = None,
    w_smooth2: Optional[Union[float, torch.Tensor]] = None,
    w_smooth3: Optional[Union[float, torch.Tensor]] = None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    device, dtype = target.device, target.dtype
    *lead_shape, num_steps = target.shape
    if w_data is None:
        w_data = torch.ones_like(target)

    A = torch.diag_embed(slope)
    weighted_A = A * w_data.unsqueeze(-1)
    ata = einops.einsum(weighted_A, A, "... i j, ... i k -> ... j k")
    rhs = einops.einsum(weighted_A, target, "... i j, ... i -> ... j")
    dtd = construct_dtd(
        num_steps,
        lead_shape,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )

    factor = None
    current_ridge = ridge
    while factor is None:
        try:
            ridge_term = current_ridge * torch.eye(num_steps, dtype=dtype, device=device).expand(
                *lead_shape, num_steps, num_steps
            )
            lhs = ata + dtd + ridge_term
            factor = torch.linalg.cholesky(lhs)
        except RuntimeError:
            current_ridge = max(current_ridge * 10, 1e-6)
    return torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def dxy_theta_to_v_without_v0(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    *lead_shape, num_steps, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy

    A = torch.zeros(*lead_shape, 2 * num_steps, num_steps + 1, dtype=dtype, device=device)
    b = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(num_steps, device=device)
    sin_rows = 2 * torch.arange(num_steps, device=device) + 1
    cols = torch.arange(num_steps, device=device)
    A[..., cos_rows, cols] = cos_theta[..., :-1]
    A[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A[..., sin_rows, cols] = sin_theta[..., :-1]
    A[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    ata = einops.einsum(A, A, "... i j, ... i k -> ... j k")
    rhs = einops.einsum(A, b, "... i j, ... i -> ... j")

    dtd = construct_dtd(
        num_steps + 1,
        lead_shape,
        device=device,
        dtype=dtype,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )
    ridge_term = v_ridge * torch.eye(num_steps + 1, dtype=dtype, device=device).expand(
        *lead_shape, num_steps + 1, num_steps + 1
    )
    factor = torch.linalg.cholesky(ata + dtd + ridge_term)
    return torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def dxy_theta_to_v(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    v0: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    *lead_shape, num_steps, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy

    A = torch.zeros(*lead_shape, 2 * num_steps, num_steps + 1, dtype=dtype, device=device)
    b = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(num_steps, device=device)
    sin_rows = 2 * torch.arange(num_steps, device=device) + 1
    cols = torch.arange(num_steps, device=device)
    A[..., cos_rows, cols] = cos_theta[..., :-1]
    A[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A[..., sin_rows, cols] = sin_theta[..., :-1]
    A[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    ata = einops.einsum(A, A, "... i j, ... i k -> ... j k")
    rhs = einops.einsum(A[..., :, 1:], b, "... i j, ... i -> ... j")
    rhs -= ata[..., 1:, 0] * v0.unsqueeze(-1)

    dtd = construct_dtd(
        num_steps + 1,
        lead_shape,
        device=device,
        dtype=dtype,
        w_smooth3=1.0,
        lam=v_lambda,
        dt=dt,
    )
    rhs -= dtd[..., 1:, 0] * v0.unsqueeze(-1)
    ridge_term = v_ridge * torch.eye(num_steps, dtype=dtype, device=device).expand(
        *lead_shape, num_steps, num_steps
    )
    factor = torch.linalg.cholesky(ata[..., 1:, 1:] + dtd[..., 1:, 1:] + ridge_term)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
    return torch.cat([v0.unsqueeze(-1), y], dim=-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
def theta_smooth(
    traj_future_rot: torch.Tensor,
    dt: float = 1.0,
    theta_lambda: float = 1e-4,
    theta_ridge: float = 1e-4,
) -> torch.Tensor:
    theta = unwrap_angle(so3_to_yaw_torch(traj_future_rot))
    theta_init = torch.zeros_like(theta[..., 0])
    return solve_single_constraint(
        x_init=theta_init,
        x_target=theta,
        w_smooth3=1.0,
        dt=dt,
        lam=theta_lambda,
        ridge=theta_ridge,
    )


class ActionSpace(ABC, nn.Module):
    @abstractmethod
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_action_space_dims(self) -> tuple[int, ...]:
        raise NotImplementedError


class UnicycleAccelCurvatureActionSpace(ActionSpace):
    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),
        curvature_bounds: tuple[float, float] = (-0.2, 0.2),
        dt: float = 0.1,
        n_waypoints: int = 64,
        theta_lambda: float = 1e-6,
        theta_ridge: float = 1e-8,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        a_lambda: float = 1e-4,
        a_ridge: float = 1e-4,
        kappa_lambda: float = 1e-4,
        kappa_ridge: float = 1e-4,
    ) -> None:
        super().__init__()
        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("curvature_mean", torch.tensor(curvature_mean))
        self.register_buffer("curvature_std", torch.tensor(curvature_std))
        self.accel_bounds = accel_bounds
        self.curvature_bounds = curvature_bounds
        self.dt = dt
        self.n_waypoints = n_waypoints
        self.theta_lambda = theta_lambda
        self.theta_ridge = theta_ridge
        self.v_lambda = v_lambda
        self.v_ridge = v_ridge
        self.a_lambda = a_lambda
        self.a_ridge = a_ridge
        self.kappa_lambda = kappa_lambda
        self.kappa_ridge = kappa_ridge

    def get_action_space_dims(self) -> tuple[int, int]:
        return (self.n_waypoints, 2)

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _v_to_a(self, velocity: torch.Tensor) -> torch.Tensor:
        delta_v = (velocity[..., 1:] - velocity[..., :-1]) / self.dt
        return solve_xs_eq_y(
            slope=torch.ones_like(delta_v),
            target=delta_v,
            dt=self.dt,
            lam=self.a_lambda,
            ridge=self.a_ridge,
            w_smooth2=1.0,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _theta_v_a_to_kappa(
        self,
        theta: torch.Tensor,
        velocity: torch.Tensor,
        accel: torch.Tensor,
    ) -> torch.Tensor:
        delta_theta = theta[..., 1:] - theta[..., :-1]
        slope = self.dt * velocity[..., :-1] + (self.dt**2) / 2.0 * accel
        return solve_xs_eq_y(
            slope=slope,
            target=delta_theta,
            w_data=torch.ones_like(delta_theta),
            w_smooth2=1.0,
            lam=self.kappa_lambda,
            ridge=self.kappa_ridge,
            dt=self.dt,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def estimate_t0_states(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        full_xy = traj_history_xyz[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = unwrap_angle(so3_to_yaw_torch(traj_history_rot))
        velocity = dxy_theta_to_v_without_v0(
            dxy=dxy,
            theta=theta,
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )
        return {"v": velocity[..., -1]}

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def traj_to_action(
        self,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        traj_future_xyz: torch.Tensor,
        traj_future_rot: torch.Tensor,
        t0_states: Optional[dict[str, torch.Tensor]] = None,
        output_all_states: bool = False,
    ) -> torch.Tensor:
        if traj_future_xyz.shape[-2] != self.n_waypoints:
            raise ValueError(
                f"future trajectory must have length {self.n_waypoints}, got {traj_future_xyz.shape[-2]}"
            )
        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)

        full_xy = torch.cat([traj_history_xyz[..., -1:, :], traj_future_xyz], dim=-2)[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = theta_smooth(
            traj_future_rot=traj_future_rot,
            dt=self.dt,
            theta_lambda=self.theta_lambda,
            theta_ridge=self.theta_ridge,
        )
        velocity = dxy_theta_to_v(
            dxy=dxy,
            theta=theta,
            v0=t0_states["v"],
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )
        accel = self._v_to_a(velocity)
        curvature = self._theta_v_a_to_kappa(theta, velocity, accel)
        accel = (accel - self.accel_mean.to(accel.device)) / self.accel_std.to(accel.device)
        curvature = (curvature - self.curvature_mean.to(curvature.device)) / self.curvature_std.to(
            curvature.device
        )
        action = torch.stack([accel, curvature], dim=-1)
        if output_all_states:
            return action, torch.stack([velocity[:, :-1], accel, theta[:, :-1]], dim=-1)
        return action


class DiscreteTrajectoryTokenizer:
    def __init__(
        self,
        action_space: ActionSpace,
        dims_min: Sequence[float],
        dims_max: Sequence[float],
        num_bins: int,
    ) -> None:
        self.action_space = action_space
        self.dims_min = list(dims_min)
        self.dims_max = list(dims_max)
        self.num_bins = num_bins

    @property
    def vocab_size(self) -> int:
        return self.num_bins

    def encode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        fut_xyz: torch.Tensor,
        fut_rot: torch.Tensor,
    ) -> torch.LongTensor:
        batch_size = fut_xyz.shape[0]
        action = self.action_space.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)
        dims_min = torch.tensor(self.dims_min, device=action.device, dtype=action.dtype)
        dims_max = torch.tensor(self.dims_max, device=action.device, dtype=action.dtype)
        action = (action - dims_min) / (dims_max - dims_min)
        action = (action * (self.num_bins - 1)).round().long().clamp(0, self.num_bins - 1)
        return action.reshape(batch_size, -1)


class BaseDiffusion(ABC, nn.Module):
    def __init__(self, x_dims: Union[Sequence[int], int]) -> None:
        super().__init__()
        self.x_dims = [x_dims] if isinstance(x_dims, int) else list(x_dims)

    @abstractmethod
    def construct_training_data(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def compute_loss_from_pred(self, training_data: dict[str, torch.Tensor], pred: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FlowMatching(BaseDiffusion):
    def __init__(
        self,
        x_dims: Union[Sequence[int], int],
        train_timestep_sampler: str = "beta",
        **_: Any,
    ) -> None:
        super().__init__(x_dims=x_dims)
        self.train_timestep_sampler = train_timestep_sampler
        if self.train_timestep_sampler == "beta":
            self.beta_dist = torch.distributions.beta.Beta(
                torch.tensor(1.5, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
            )
            self.beta_scale_constant = 0.999

    def construct_training_data(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        if self.train_timestep_sampler == "uniform":
            timesteps = torch.rand((batch_size,), device=x.device)
        elif self.train_timestep_sampler == "beta":
            timesteps = self.beta_dist.sample((batch_size,)).to(x.device)
            timesteps = self.beta_scale_constant - timesteps * self.beta_scale_constant
        else:
            raise ValueError(f"Unsupported timestep sampler: {self.train_timestep_sampler}")

        while timesteps.ndim < x.ndim:
            timesteps = timesteps.unsqueeze(-1)
        noise = torch.randn_like(x)
        noisy_x = timesteps * x + (1 - timesteps) * noise
        return {"x": x, "noisy_x": noisy_x, "timesteps": timesteps, "noise": noise}

    def compute_loss_from_pred(self, training_data: dict[str, torch.Tensor], pred: torch.Tensor) -> torch.Tensor:
        target = (training_data["x"] - training_data["noise"]).to(dtype=pred.dtype)
        return torch.nn.functional.mse_loss(target, pred)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.type_as(x) * self.weight


class MLPEncoder(nn.Module):
    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, out_dim: int):
        super().__init__()
        if num_enc_layers < 1:
            raise ValueError("num_enc_layers must be >= 1")
        layers = [nn.Linear(num_input_feats, hidden_size), nn.SiLU()]
        for layer_index in range(num_enc_layers):
            if layer_index < num_enc_layers - 1:
                layers.extend(
                    [RMSNorm(hidden_size, eps=1e-5), nn.Linear(hidden_size, hidden_size), nn.SiLU()]
                )
            else:
                layers.extend([RMSNorm(hidden_size, eps=1e-5), nn.Linear(hidden_size, out_dim)])
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class FourierEncoderV2(nn.Module):
    def __init__(self, dim: int, max_freq: float = 100.0):
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1) * math.sqrt(2)


class PerWaypointActionInProjV2(nn.Module):
    def __init__(
        self,
        in_dims: Sequence[int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ) -> None:
        super().__init__()
        self.sinus = nn.ModuleList(
            [FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq) for _ in range(in_dims[-1])]
        )
        self.timestep_fourier_encoder = FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
        num_input_feats = sum(encoder.out_dim for encoder in self.sinus) + self.timestep_fourier_encoder.out_dim
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            out_dim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, num_waypoints, _ = x.shape
        x = x.float()
        timesteps = timesteps.float()
        action_feats = torch.cat([encoder(x[:, :, i]) for i, encoder in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1]).repeat(1, num_waypoints, 1)
        hidden = torch.cat((action_feats, timestep_feats), dim=-1)
        hidden = self.encoder(hidden.flatten(0, 1)).reshape(batch_size, num_waypoints, -1)
        return self.norm(hidden)


def _strip_meta_keys(config: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if config is None:
        return None
    cfg = deepcopy(config)
    for key in [k for k in cfg if k.startswith("_")]:
        cfg.pop(key, None)
    return cfg


def _normalize_action_space_config(config: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    cfg = _strip_meta_keys(config)
    if cfg is None:
        return None
    target = str(config.get("_target_", "")).lower() if config is not None else ""
    kind = cfg.pop("type", None)
    if "unicycleaccelcurvatureactionspace" in target:
        kind = "unicycle_accel_curvature"
    if kind is None:
        kind = "unicycle_accel_curvature"
    cfg["type"] = kind
    return cfg


def _normalize_discrete_tokenizer_config(config: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    cfg = _strip_meta_keys(config)
    if cfg is None:
        return None
    target = str(config.get("_target_", "")).lower() if config is not None else ""
    kind = cfg.pop("type", None)
    if "discretetrajectorytokenizer" in target:
        kind = "discrete"
    if kind is None:
        kind = "discrete"
    action_space_cfg = cfg.pop("action_space", None) or cfg.pop("action_space_cfg", None)
    cfg["action_space"] = _normalize_action_space_config(action_space_cfg)
    cfg["type"] = kind
    return cfg


def _normalize_diffusion_config(config: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    cfg = _strip_meta_keys(config)
    if cfg is None:
        return None
    target = str(config.get("_target_", "")).lower() if config is not None else ""
    kind = cfg.pop("type", None)
    if "flowmatching" in target:
        kind = "flow_matching"
    if kind is None:
        kind = "flow_matching"
    cfg["type"] = kind
    return cfg


def _normalize_action_in_proj_config(config: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    cfg = _strip_meta_keys(config)
    if cfg is None:
        return None
    target = str(config.get("_target_", "")).lower() if config is not None else ""
    kind = cfg.pop("type", None)
    if "perwaypointactioninprojv2" in target:
        kind = "per_waypoint_v2"
    if kind is None:
        kind = "per_waypoint_v2"
    cfg["type"] = kind
    return cfg


def _normalize_action_out_proj_config(config: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    cfg = _strip_meta_keys(config) or {}
    target = str(config.get("_target_", "")).lower() if config is not None else ""
    kind = cfg.pop("type", None)
    if "linear" in target:
        kind = "linear"
    if kind is None:
        kind = "linear"
    cfg["type"] = kind
    return cfg


def build_action_space(config: Optional[dict[str, Any]]) -> ActionSpace:
    cfg = _normalize_action_space_config(config)
    if cfg is None:
        raise ValueError("`action_space_cfg` configuration is required.")
    kind = cfg.pop("type", "unicycle_accel_curvature")
    if kind != "unicycle_accel_curvature":
        raise ValueError(f"Unsupported action space type: {kind}")
    return UnicycleAccelCurvatureActionSpace(**cfg)


def build_trajectory_tokenizer(config: Optional[dict[str, Any]]) -> Optional[DiscreteTrajectoryTokenizer]:
    cfg = _normalize_discrete_tokenizer_config(config)
    if cfg is None:
        return None
    kind = cfg.pop("type", "discrete")
    if kind != "discrete":
        raise ValueError(f"Unsupported trajectory tokenizer type: {kind}")
    action_space = build_action_space(cfg.pop("action_space"))
    return DiscreteTrajectoryTokenizer(action_space=action_space, **cfg)


def build_diffusion(config: Optional[dict[str, Any]], x_dims: Sequence[int]) -> BaseDiffusion:
    cfg = _normalize_diffusion_config(config) or {"type": "flow_matching"}
    kind = cfg.pop("type", "flow_matching")
    if kind != "flow_matching":
        raise ValueError(f"Unsupported diffusion type: {kind}")
    return FlowMatching(x_dims=x_dims, **cfg)


def build_action_input_projector(
    config: Optional[dict[str, Any]],
    in_dims: Sequence[int],
    out_dim: int,
) -> nn.Module:
    cfg = _normalize_action_in_proj_config(config) or {"type": "per_waypoint_v2"}
    kind = cfg.pop("type", "per_waypoint_v2")
    if kind != "per_waypoint_v2":
        raise ValueError(f"Unsupported action input projector type: {kind}")
    return PerWaypointActionInProjV2(in_dims=in_dims, out_dim=out_dim, **cfg)


def build_action_output_projector(
    config: Optional[dict[str, Any]],
    in_features: int,
    out_features: int,
) -> nn.Module:
    cfg = _normalize_action_out_proj_config(config)
    kind = cfg.pop("type", "linear")
    if kind != "linear":
        raise ValueError(f"Unsupported action output projector type: {kind}")
    return nn.Linear(in_features, out_features, **cfg)


__all__ = [
    "ActionSpace",
    "BaseDiffusion",
    "DiscreteTrajectoryTokenizer",
    "FlowMatching",
    "PerWaypointActionInProjV2",
    "build_action_input_projector",
    "build_action_output_projector",
    "build_action_space",
    "build_diffusion",
    "build_trajectory_tokenizer",
]
