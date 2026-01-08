import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from FrEIA.modules import BinnedSplineBase, rational_quadratic_spline


class ElementwiseRationalQuadraticSpline(BinnedSplineBase):
    def __init__(self, bins: int = 10, **kwargs) -> None:
        super().__init__([(-1,)], None, bins=bins, parameter_counts={"deltas": bins - 1}, **kwargs)
        self.num_params = sum(self.parameter_counts.values())

    def constrain_parameters(self, parameters: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        parameters = super().constrain_parameters(parameters)
        # we additionally want positive derivatives to preserve monotonicity
        # the derivative must also match the tails at the spline boundaries
        deltas = parameters["deltas"]

        # shifted softplus such that network output 0 -> delta = scale
        shift = np.log(np.e - 1)
        deltas = F.softplus(deltas + shift)

        # boundary condition: derivative is equal to affine scale at spline boundaries
        scale = torch.sum(parameters["heights"], dim=-1, keepdim=True) / torch.sum(parameters["widths"], dim=-1,
                                                                                   keepdim=True)
        scale = scale.expand(*scale.shape[:-1], 2)

        deltas = torch.cat((deltas, scale), dim=-1).roll(1, dims=-1)

        parameters["deltas"] = deltas

        return parameters

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        return input_dims

    def forward(self, x_or_z: torch.Tensor, parameters,
                rev: bool = False, jac: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        parameters = self.split_parameters(parameters, x_or_z.shape[1])
        parameters = self.constrain_parameters(parameters)

        return self.binned_spline(x=x_or_z, parameters=parameters, spline=self.spline, rev=rev)

    def spline(self, x: torch.Tensor, parameters: Dict[str, torch.Tensor], rev: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor]:
        left, right, bottom, top = parameters["left"], parameters["right"], parameters["bottom"], parameters["top"]
        deltas_left, deltas_right = parameters["deltas_left"], parameters["deltas_right"]
        return rational_quadratic_spline(x, left, right, bottom, top, deltas_left, deltas_right, rev=rev)

    def binned_spline(self, x: torch.Tensor, parameters: Dict[str, torch.Tensor], spline: callable,
                      rev: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the spline for given bin and spline parameters.

        Copy from FrEIA to get per-dimension volume change (avoids overflow -> mean instead of sum).
        """
        x = x.movedim(1, -1)

        # find bin knots
        knot_x = parameters["left"] + torch.cumsum(parameters["widths"], dim=-1)
        knot_y = parameters["bottom"] + torch.cumsum(parameters["heights"], dim=-1)

        # concatenate leftmost edge
        knot_x = torch.cat((parameters["left"], knot_x), dim=-1)
        knot_y = torch.cat((parameters["bottom"], knot_y), dim=-1)

        # find spline mask
        if not rev:
            inside = (knot_x[..., 0] < x) & (x <= knot_x[..., -1])
        else:
            y = x
            inside = (knot_y[..., 0] < y) & (y <= knot_y[..., -1])

        knot_x = knot_x[inside]
        knot_y = knot_y[inside]

        x_in = x[inside]
        x_out = x[~inside]

        scale = torch.sum(parameters["heights"], dim=-1, keepdim=True) / torch.sum(parameters["widths"], dim=-1,
                                                                                   keepdim=True)
        shift = parameters["bottom"] - scale * parameters["left"]

        scale = scale[~inside].squeeze(-1)
        shift = shift[~inside].squeeze(-1)

        # find bin edge indices
        if not rev:
            upper = torch.searchsorted(knot_x, x_in[..., None])
        else:
            y_in = x_in
            upper = torch.searchsorted(knot_y, y_in[..., None])
        lower = upper - 1

        spline_parameters = dict()

        # gather bin edges from indices
        spline_parameters["left"] = torch.gather(knot_x, dim=-1, index=lower).squeeze(-1)
        spline_parameters["right"] = torch.gather(knot_x, dim=-1, index=upper).squeeze(-1)
        spline_parameters["bottom"] = torch.gather(knot_y, dim=-1, index=lower).squeeze(-1)
        spline_parameters["top"] = torch.gather(knot_y, dim=-1, index=upper).squeeze(-1)

        # gather all other parameter edges
        for key, value in parameters.items():
            if key in ["left", "bottom", "widths", "heights", "total_width"]:
                continue

            v = value[inside]

            spline_parameters[f"{key}_left"] = torch.gather(v, dim=-1, index=lower).squeeze(-1)
            spline_parameters[f"{key}_right"] = torch.gather(v, dim=-1, index=upper).squeeze(-1)

        if not rev:
            y = torch.clone(x)
            log_jac = y.new_zeros(y.shape)

            y[inside], log_jac[inside] = spline(x_in, spline_parameters, rev=rev)
            y[~inside], log_jac[~inside] = scale * x_out + shift, torch.log(scale).to(log_jac)

            y = y.movedim(-1, 1)

            return y, log_jac
        else:
            y_in = x_in
            y_out = x_out

            x_inside, log_jac_inside = spline(y_in, spline_parameters, rev=rev)
            x = torch.zeros(x.shape, dtype=x_inside.dtype, device=x_inside.device)
            log_jac = torch.zeros(x.shape, dtype=log_jac_inside.dtype, device=log_jac_inside.device)
            x[inside], log_jac[inside] = x_inside, log_jac_inside
            x[~inside], log_jac[~inside] = (y_out - shift) / scale, torch.log(scale)

            x = x.movedim(-1, 1)

            return x, -log_jac


class Flow1D(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.spline = ElementwiseRationalQuadraticSpline(*args, **kwargs)
        self.entropy_1d = math.log(2 * math.pi) / 2

    @property
    def num_params(self):
        return self.spline.num_params
    
    def compute_log_prob_and_z(self, x: torch.Tensor, parameters, is_log: bool = False,
                               log_epsilon=1e-5, is_fit_gaussian: torch.Tensor | bool | None = None):
        if is_fit_gaussian is None or is_fit_gaussian is False:
            is_fit_gaussian = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        elif is_fit_gaussian is True:
            is_fit_gaussian = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        else:
            raise ValueError(f"is_fit_gaussian must be a boolean tensor, True, False, or None, got: {is_fit_gaussian!r}")
        if is_log:
            x = torch.log(x + log_epsilon)
            log_jac = -x
        else:
            log_jac = 0
        z = torch.zeros_like(x)
        log_prob = torch.zeros_like(x)

        if is_fit_gaussian.any():
            samples = torch.stack([
                self.sample(parameters[is_fit_gaussian])
                for _ in range(100)
            ])
            mean = samples.mean(dim=0)
            std = samples.std(dim=0)
            z[is_fit_gaussian] = (x[is_fit_gaussian] - mean) / std
            log_prob[is_fit_gaussian] = -0.5 * ((x[is_fit_gaussian] - mean) / std).pow(2) - std.log() - self.entropy_1d

        if (~is_fit_gaussian).any():
            z_spline, log_jac_spline = self.spline(x[:, None], parameters)
            log_jac = log_jac + log_jac_spline[:, 0]
            log_prob[~is_fit_gaussian] = -z_spline[:, 0] ** 2 / 2 - self.entropy_1d + log_jac
            z[~is_fit_gaussian] = z_spline[:, 0]

        return log_prob, z

    def forward(self, x: torch.Tensor, parameters, is_log: bool=False,
                is_fit_gaussian: torch.Tensor | bool | None = None) -> torch.Tensor:
        log_prob, _ = self.compute_log_prob_and_z(x, parameters, is_log, is_fit_gaussian=is_fit_gaussian)
        return log_prob

    def intensity(self, x: torch.Tensor, parameters: torch.Tensor, is_log: bool = False, log_epsilon=1e-5) -> torch.Tensor:
        log_prob, z = self.compute_log_prob_and_z(x, parameters, is_log, log_epsilon)
        prob = log_prob.exp()
        cdf = torch.distributions.Normal(0, 1, validate_args=False).cdf(z)
        intensity = prob / (1 - cdf)
        return intensity

    def sample(self, parameters, median=False) -> torch.Tensor:
        if median:
            z = torch.zeros_like(parameters[:, 0])
        else:
            z = torch.randn_like(parameters[:, 0])
        x, _ = self.spline(z[:, None], parameters, rev=True)
        return x[:, 0]


class GMM1D(torch.nn.Module):
    def __init__(self, num_components: int, dropout: float = 0.0, fix_std: bool = False, bins=None):
        assert bins is None, "bins needs to be None for GMM, it only exists for compatibility"
        super().__init__()
        self.num_components = num_components
        self.fix_std = fix_std
        self.num_params = num_components * (2 if fix_std else 3)
        self.dropout = dropout

    def extract_gmm(self, parameters):
        weights = parameters[:, :self.num_components].log_softmax(dim=-1)
        means = parameters[:, self.num_components:2 * self.num_components]
        if self.fix_std:
            stds = 1 + 0 * weights
        else:
            stds = parameters[:, 2 * self.num_components:].exp()
        if self.training and self.dropout > 0:
            remove_component = torch.rand_like(weights) < self.dropout
            weights, means, stds = [
                entry[~remove_component]
                for entry in (weights, means, stds)
            ]
        return weights, means, stds

    def forward(self, x: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        # Expand dimensions for broadcasting: (batch_size, num_components)
        weights, means, stds = self.extract_gmm(parameters)
        x = x.unsqueeze(-1)  # (batch_size, 1)

        # Compute log probability of each component
        component_log_probs = -0.5 * (((x - means) / stds) ** 2 + 2 * stds.log() + math.log(2 * torch.pi))
        return torch.logsumexp(weights + component_log_probs, dim=-1)  # (batch_size)

    def sample(self, parameters, median=False) -> torch.Tensor:
        weights, means, stds = self.extract_gmm(parameters)
        # Sample from weights
        components = torch.multinomial(weights.exp(), 1)[:, 0]
        batch_idx = torch.arange(parameters.shape[0], device=parameters[0].device)
        if median:
            assert self.num_components == 1, "Median is only supported for one component"
            return means[batch_idx, components]
        noise = torch.randn_like(means[batch_idx, components])
        return means[batch_idx, components] + stds[batch_idx, components] * noise
