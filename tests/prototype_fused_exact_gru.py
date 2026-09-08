"""Experimental inference-only GRU comparison; never imported by production.

This intentionally remains a test utility. Equal equations and complete
histories do not imply bitwise equality between GRUCell and sequence kernels.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
from tensordict import TensorDict
from torch import nn

from active_adaptation.learning.ppo.ppo_vel import (
    GRU,
    _EXACT_RECURRENT_LENGTHS,
    exact_recurrent_lengths,
    recurrent_mode,
    set_recurrent_mode,
)
from benchmark_exact_online_encoder import BenchmarkPolicy


class ExperimentalFusedGRU(nn.Module):
    """Use identical PyTorch GRU gates/weights through a sequence kernel.

    Raw state is selected before LayerNorm at the last *real* timestep.
    Interior resets and gradient-enabled calls are rejected, not approximated.
    Existing weight Parameters are shared; no copied weight can become stale.
    """

    def __init__(self, original: GRU):
        super().__init__()
        cell = original.gru
        with torch.random.fork_rng(devices=[]):
            fused = nn.GRU(cell.input_size, cell.hidden_size, num_layers=1, batch_first=True)
        for source, destination in (
            ("weight_ih", "weight_ih_l0"), ("weight_hh", "weight_hh_l0"),
            ("bias_ih", "bias_ih_l0"), ("bias_hh", "bias_hh_l0"),
        ):
            setattr(fused, destination, getattr(cell, source))
        self.fused = fused.eval()
        self.ln = original.ln

    def forward(self, x, is_init, hx):
        if torch.is_grad_enabled() or not recurrent_mode():
            raise RuntimeError("Experimental fused GRU only supports no-grad recurrent inference")
        if x.ndim != 3 or hx.ndim != 3 or hx.shape[:2] != x.shape[:2]:
            raise ValueError("Experimental fused GRU requires aligned [N,T] input and hidden tensors")
        if is_init.dtype != torch.bool:
            raise TypeError("Experimental fused GRU requires boolean reset indicators")
        n, time = x.shape[:2]
        resets = is_init.reshape(n, time, 1)
        # Deliberate validation cost in this experimental implementation. Do
        # not silently run a no-reset fused kernel on a reset-containing chunk.
        if bool(resets[:, 1:].any()):
            raise RuntimeError("Experimental fused GRU rejects interior resets")
        initial = hx[:, 0] * (1.0 - resets[:, 0].to(hx.dtype))
        raw, final = self.fused(x, initial.unsqueeze(0))
        lengths = _EXACT_RECURRENT_LENGTHS.get()
        if lengths is None:
            final = final[0]
        else:
            lengths_cpu, lengths_device = lengths
            if len(lengths_cpu) != n or max(lengths_cpu) > time or lengths_device.device != x.device:
                raise ValueError("Experimental fused GRU lengths must match the [N,T] input")
            final = raw.gather(
                1, (lengths_device - 1).view(n, 1, 1).expand(n, 1, raw.shape[-1]),
            ).squeeze(1)
        return self.ln(raw), final.unsqueeze(1).expand(n, time, -1)


def experimental_fused_copy(policy):
    """Return an isolated candidate; leave the supplied production model alone."""
    candidate = copy.deepcopy(policy)

    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, GRU):
                setattr(module, name, ExperimentalFusedGRU(child))
            else:
                replace(child)

    replace(candidate)
    return candidate.eval().requires_grad_(False)


@contextmanager
def comparison_precision(*, strict_fp32=False, cpu_threads=1):
    """Only the experiment changes precision, restoring process settings."""
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(cpu_threads)
        if strict_fp32:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_num_threads(threads)


def build_policy_pair(*, device="cpu", residual=True):
    # BenchmarkPolicy builds the actual production 36x64 depth CNN, object
    # transforms and both GRUs, with a live nonzero depth residual projection.
    policy = BenchmarkPolicy(device=device, latent_dim=256)
    if not residual:
        # Construct the actual ordinary GRUModule route, including its
        # three-input TensorDictModule signature.
        from test_tvkd_depth_residual import build_full_ppovel
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1943)
            source = build_full_ppovel(residual=False, latent_dim=256)
        policy.cfg.perception_depth_residual = False
        for name in ("temporal_depth_gru_ema", "object_adapt_ema", "object_pred_transform", "adapt_ema"):
            setattr(policy, name, getattr(source, name).to(policy.device))
        policy.eval().requires_grad_(False)
    return policy, experimental_fused_copy(policy)


def inputs(policy, *, time=36):
    generator = torch.Generator().manual_seed(51319)
    n = 3
    values = {
        "depth": torch.randint(0, 101, (n, time, 1, 36, 64), generator=generator).float() / 100,
        "policy": torch.randn(n, time, 10, generator=generator),
        "vel_command": torch.randn(n, time, 5, generator=generator),
        "object_geo_": policy.geometry[:n, None].expand(n, time, -1).clone(),
        "is_init": torch.zeros(n, time, 1, dtype=torch.bool),
        "depth_hx": torch.zeros(n, time, policy.depth_feature_dim),
        "adapt_hx": torch.zeros(n, time, policy.cfg.latent_dim),
    }
    values["is_init"][:, 0] = True
    return TensorDict(values, [n, time]).to(policy.device)


def run_graph(policy, td, lengths=None):
    with torch.no_grad(), set_recurrent_mode(True):
        if lengths is None:
            return policy.run_stack(td.clone())
        with exact_recurrent_lengths(lengths, torch.tensor(lengths, device=policy.device)):
            return policy.run_stack(td.clone())


def comparison_outputs(td, lengths=None):
    if lengths is None:
        latent = td["priv_pred"].flatten(0, 1)
    else:
        latent = torch.cat([td["priv_pred"][row, :length] for row, length in enumerate(lengths)])
    return {
        "priv_pred": latent.detach().cpu(),
        "depth_hx": td["next", "depth_hx"][:, -1].detach().cpu(),
        "adapt_hx": td["next", "adapt_hx"][:, -1].detach().cpu(),
    }


def comparison_report(reference, candidate):
    report = {}
    for key in reference:
        baseline, result = reference[key], candidate[key]
        difference = (result - baseline).abs()
        entry = {
            "torch_equal": torch.equal(result, baseline),
            "max_absolute_difference": difference.max().item(),
            "max_relative_difference": (difference / baseline.abs().clamp_min(torch.finfo(baseline.dtype).tiny)).max().item(),
            "default_assert_close_passed": True,
        }
        try:
            torch.testing.assert_close(result, baseline)
        except AssertionError as error:
            entry["default_assert_close_passed"] = False
            entry["default_assert_close_error"] = str(error)
        report[key] = entry
    return report


def compare_full_pipeline(policy, candidate):
    raw = inputs(policy)
    baseline_full = run_graph(policy, raw)
    candidate_full = run_graph(candidate, raw)
    reports = {"full_reset": comparison_report(
        comparison_outputs(baseline_full), comparison_outputs(candidate_full),
    )}
    prefix_lengths = [1, 31, 32]
    baseline_prefix = run_graph(policy, raw[:, :32], prefix_lengths)
    candidate_prefix = run_graph(candidate, raw[:, :32], prefix_lengths)
    reports["padded_prefix"] = comparison_report(
        comparison_outputs(baseline_prefix, prefix_lengths),
        comparison_outputs(candidate_prefix, prefix_lengths),
    )
    suffix_lengths = [4, 3, 1]
    suffix_data = raw[:, :4].clone()
    for row, (start, length) in enumerate(zip(prefix_lengths, suffix_lengths)):
        for key in suffix_data.keys():
            suffix_data[key][row].zero_()
            suffix_data[key][row, :length] = raw[key][row, start:start + length]
    baseline_suffix_input = suffix_data.clone()
    candidate_suffix_input = suffix_data.clone()
    for name in ("depth_hx", "adapt_hx"):
        baseline_suffix_input[name] = baseline_prefix["next", name][:, -1:].expand_as(suffix_data[name]).clone()
        candidate_suffix_input[name] = candidate_prefix["next", name][:, -1:].expand_as(suffix_data[name]).clone()
    baseline_suffix = run_graph(policy, baseline_suffix_input, suffix_lengths)
    candidate_suffix = run_graph(candidate, candidate_suffix_input, suffix_lengths)
    reports["continued_suffix"] = comparison_report(
        comparison_outputs(baseline_suffix, suffix_lengths),
        comparison_outputs(candidate_suffix, suffix_lengths),
    )
    return reports
