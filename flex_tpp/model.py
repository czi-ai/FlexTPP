from collections import OrderedDict
from typing import Tuple, Callable

import torch
from hydra.utils import instantiate
from torch import Tensor
from math import ceil
from torch.nn.functional import cross_entropy
from tqdm.auto import trange

from flex_tpp.data.base import MODALITY_CONTINUOUS, MODALITY_CATEGORICAL, MODALITY_ORDINAL, MODALITY_GAUSSIAN, \
    BatchSpec, \
    partial_batch
from flex_tpp.invertible_1d import Flow1D
from flex_tpp.mha import MultiHeadAttention
import math

SAVE_NUM_CLASSES = {
    torch.bfloat16: 512,
    torch.float16: 2048,
    torch.float32: 16_777_216,
    torch.float64: 2 ** 53
}


class Transformer(torch.nn.Module):
    def __init__(self, dim_m: int, dim_k, dim_ff, depth,
                 normalize=False, non_linearity="Softplus",
                 dropout: float = 0.0, is_causal: bool = True):
        if dim_m % dim_k != 0:
            raise ValueError(f"dim_m must be divisible by dim_k, but {dim_m=} and {dim_k=}")
        super().__init__()
        self.dim_m = dim_m
        self.depth = depth
        self.normalize = normalize
        for idx in range(depth):
            if normalize:
                self.add_module(f"attn_norm_{idx}", torch.nn.LayerNorm(dim_m))
            self.add_module(f"block_{idx}", MultiHeadAttention(
                embed_dim=dim_m, num_heads=dim_m // dim_k, dropout=dropout,
                is_causal=is_causal
            ))
            if normalize:
                self.add_module(f"ff_norm_{idx}", torch.nn.LayerNorm(dim_m))
            self.add_module(f"ff_{idx}", torch.nn.Sequential(OrderedDict({
                "0": torch.nn.Linear(dim_m, dim_ff),
                "1": getattr(torch.nn, non_linearity)(),
                "dropout": torch.nn.Dropout(dropout),
                "2": torch.nn.Linear(dim_ff, dim_m),
            })))

    def forward(self, x):
        for idx in range(self.depth):
            before_attn = x
            if self.normalize:
                x = getattr(self, f"attn_norm_{idx}")(x)
            x = getattr(self, f"block_{idx}")(x, x, x)
            x = before_attn + x

            before_ff = x
            if self.normalize:
                x = getattr(self, f"ff_norm_{idx}")(x)
            x = getattr(self, f"ff_{idx}")(x)
            x = before_ff + x
        return x


class EmbeddingTransformer(Transformer):
    def __init__(self, dim_m, dropout, window_size: int = None, positional_encoding_migrated=False, **kwargs):
        super().__init__(dim_m=dim_m, dropout=dropout, **kwargs)
        self.window_size = dim_m if window_size is None else window_size
        self.position_encoding = PositionalEncoding(dim_m, dropout, include_input=not positional_encoding_migrated)
        if window_size != dim_m:
            self.project = torch.nn.Linear(self.window_size, self.dim_m)
        else:
            self.project = torch.nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_points = x.shape
        # Reshape x into (batch_size, N, window_size) where N = ceil(x.shape[1] / window_size)
        seq_len = ceil(num_points / self.window_size)
        pad_len = seq_len * self.window_size - num_points
        if pad_len > 0:
            pad = x.new_zeros(batch_size, pad_len)
            x = torch.cat([x, pad], dim=1)
        x = x.view(batch_size, seq_len, self.window_size)
        x = self.project(x)
        return x + self.position_encoding(x)


def make_param_net(dim_in, n_hidden_layers, hidden_unit_factor, dims_out):
    if n_hidden_layers == 0:
        return torch.nn.Linear(dim_in, dims_out)
    n_hidden_units = hidden_unit_factor * max(dim_in, dims_out)
    net = [
        torch.nn.Linear(dim_in, n_hidden_units),
        torch.nn.ReLU()
    ]
    for idx in range(1, n_hidden_layers - 1):
        net.append(torch.nn.Linear(n_hidden_units, n_hidden_units))
        net.append(torch.nn.ReLU())
    net.append(torch.nn.Linear(n_hidden_units, dims_out))
    return torch.nn.Sequential(*net)


class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000, lookup_mode=False, include_input=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("positional_encoding", pe)

        self.lookup_mode = lookup_mode
        self.include_input = include_input

    def forward(self, x):
        if self.lookup_mode:
            return self.dropout(self.positional_encoding[0, x])
        else:
            positional_encoding = self.positional_encoding[:, :x.size(1)].requires_grad_(False)
            if self.include_input:
                positional_encoding = x + positional_encoding
            return self.dropout(positional_encoding)


class FlexTPPModel(torch.nn.Module):
    def __init__(self, dim: int, dim_m: int, dim_k, dim_ff, depth,
                 normalize=False, non_linearity="Softplus",
                 conditioning_network: dict = None,
                 monotonic_bins: int = 128, dropout: float = 0.0,
                 monotonic_network: dict = None,
                 param_nets_n_hidden_layer: int = 1,
                 param_nets_hidden_dim_factor: int = 2,
                 max_num_classes: int = 2):
        assert dim_m % dim_k == 0, "Dimensions must be a multiple of the number of keys."
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.dim_m = dim_m
        self.dropout = dropout
        self.normalize = normalize

        # Linear function with bias to encode continuous data
        self.cont_embedding = torch.nn.Linear(1, dim_m)
        # One embedding vector for each class
        if max_num_classes <= 0:
            raise ValueError("max_num_classes must be > 0 (even if your data is always continuous, it won't hurt).")
        self.disc_embedding = torch.nn.Embedding(max_num_classes, dim_m)
        self.max_num_classes = max_num_classes
        # One embedding vector for each dimension
        self.dim_embedding = torch.nn.Embedding(dim, dim_m)
        # One embedding vector to inform about the expected output type
        self.type_embedding = torch.nn.Embedding(3, dim_m)
        # Inform about nan values
        self.nan_placeholder = torch.nn.Parameter(torch.randn(dim_m))

        # For handling input
        if conditioning_network is not None:
            if "dim_k" in conditioning_network and conditioning_network["dim_k"] is None:
                print(f"Setting {dim_k=}")
                conditioning_network["dim_k"] = dim_k
            self.conditioning_network = instantiate(conditioning_network, dim_m=dim_m)

        # Main predictor
        self.transformer = Transformer(
            dim_m, dim_k, dim_ff, depth,
            normalize=normalize, non_linearity=non_linearity, dropout=dropout
        )

        # Continuous distribution
        if monotonic_network is not None:
            assert monotonic_bins == 128, f"Do not overwrite monotonic_bins if specifying monotonic_network, found {monotonic_bins}."
            self.monotonic_network = instantiate(monotonic_network)
        else:
            self.monotonic_network = Flow1D(monotonic_bins)
        self.cont_parameter_net = make_param_net(
            dim_m,
            param_nets_n_hidden_layer, param_nets_hidden_dim_factor,
            self.monotonic_network.num_params
        )
        # This stabilizes training, equally spacing the splines regardless of input
        self.cont_parameter_net[-1].weight.data.fill_(0.0)
        self.cont_parameter_net[-1].bias.data.fill_(0.0)

        # Categorical distribution
        self.disc_parameter_net = make_param_net(
            dim_m,
            param_nets_n_hidden_layer, param_nets_hidden_dim_factor,
            max_num_classes
        )

        # Ordinal distribution
        self.ordinal_parameter_net = make_param_net(
            dim_m,
            param_nets_n_hidden_layer, param_nets_hidden_dim_factor,
            max_num_classes
        )

    def embed_condition(self, condition):
        return self.conditioning_network(condition)

    def embed_data(self, x_in, x_modality, condition=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed depending on if a data point is continuous or discrete
        # Shift the input by one dimension
        x_shifted = torch.cat([
            torch.zeros_like(x_in[:, :1]), x_in[:, :-1]
        ], dim=-1)
        modality_shifted = torch.cat([
            torch.full_like(x_modality[:, :1], MODALITY_CATEGORICAL, dtype=torch.long),
            x_modality[:, 1:]
        ], dim=-1)
        x = self.nan_placeholder[None, None, :].to(x_in).repeat(*x_in.shape, 1)
        for modality, embedding_fn in {
            MODALITY_CONTINUOUS: lambda x_cont: self.cont_embedding(x_cont[:, None]),
            MODALITY_GAUSSIAN: lambda x_cont: self.cont_embedding(x_cont[:, None]),
            MODALITY_CATEGORICAL: lambda x_disc: self.disc_embedding(x_disc.clamp(0, self.max_num_classes - 1).long()),
            MODALITY_ORDINAL: lambda x_ord: self.disc_embedding(x_ord.clamp(0, self.max_num_classes - 1).long())
        }.items():
            modality_mask: torch.Tensor = (modality_shifted == modality) & torch.isfinite(x_shifted)
            if torch.any(modality_mask):
                x[modality_mask] = embedding_fn(x_shifted[modality_mask]).to(x)

        # Add condition before
        if condition is not None:
            embedded_condition = self.embed_condition(condition)
        else:
            embedded_condition = torch.empty(x.shape[0], 0, self.dim_m, dtype=x.dtype, device=x.device)
        return x, embedded_condition

    def embed_position(self, batch):
        return self.dim_embedding(torch.arange(batch.data.shape[1], device=batch.data.device, dtype=torch.long)[None])

    def embed(self, batch: BatchSpec) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_in = batch.data
        modality = batch.types
        condition = batch.condition
        x, embedded_condition = self.embed_data(x_in, modality, condition)
        # Inform about what kind of parameter to predict and what dimension this is
        return x_in, torch.cat([
            embedded_condition,
            x
            + self.type_embedding(torch.where(
                modality.long() == MODALITY_GAUSSIAN,
                MODALITY_CONTINUOUS,
                modality.long()
            ))
            + self.embed_position(batch)
        ], 1), modality

    def log_prob_functions(self):
        return {
            MODALITY_CONTINUOUS: (
                self.cont_parameter_net, lambda x_cont, params: self.monotonic_network(x_cont, params)
            ),
            MODALITY_CATEGORICAL: (self.disc_parameter_net, lambda x_cat, params: -cross_entropy(params, x_cat.long())),
            MODALITY_GAUSSIAN: (
                self.cont_parameter_net,
                lambda x_cont, params: self.monotonic_network(x_cont, params, is_fit_gaussian=True)
            ),
        }

    def log_prob(self, batch: BatchSpec, return_parameters=False, return_masks=False):
        x_in, x_embedded, x_modality = self.embed(batch)
        if len(x_modality.shape) != 2:
            raise ValueError("is_discrete must be None or a 2-dimensional tensor.")
        if SAVE_NUM_CLASSES[x_in.dtype] < self.max_num_classes:
            raise ValueError(
                f"The number of allowed classes for dtype {x_in.dtype} cannot be exceed the safe number of classes."
            )

        # batch size x (condition + sequence length) x embedding dimension
        # -> batch size x sequence length x embedding dimension
        x = self.transformer(x_embedded)[:, -x_in.shape[1]:]

        log_probs = torch.full_like(x_in, float('nan'))
        if batch.log_prob_correction is not None:
            log_prob_correction = batch.log_prob_correction
        else:
            log_prob_correction = torch.zeros_like(log_probs)
        parameters = {}
        masks = {}
        for modality, (param_net, log_prob_fn) in self.log_prob_functions().items():
            # noinspection PyTypeChecker
            modality_mask: torch.Tensor = (x_modality == modality) & torch.isfinite(x_in)
            masks[modality] = modality_mask
            if torch.any(modality_mask):
                parameters[modality] = param_net(x[modality_mask])
                log_probs[modality_mask] = log_prob_fn(x_in[modality_mask], parameters[modality]) + log_prob_correction[
                    modality_mask]

        if return_parameters:
            if return_masks:
                return log_probs, parameters, masks
            return log_probs, parameters
        elif return_masks:
            raise ValueError("Cannot return masks without returning parameters.")
        return log_probs

    def sample_functions(self, arg_max: bool = False, mean_of: int = 1, median=False):
        if median:
            assert mean_of == 1, "Only take the median of one value"
        return {
            MODALITY_CONTINUOUS: lambda params: torch.stack([
                self.monotonic_network.sample(params, median=median)
                for _ in range(mean_of)
            ], dim=-1).mean(dim=-1),
            MODALITY_CATEGORICAL: lambda params: (
                params.argmax(dim=-1).float()
                if arg_max else
                torch.multinomial(params.softmax(dim=-1), 1)[:, 0]
            ),
            MODALITY_GAUSSIAN: lambda params: torch.stack([
                self.monotonic_network.sample(params, median=median)
                for _ in range(max(mean_of, 100))
            ], dim=-1).mean(dim=-1)
        }

    def _single_dim_sample(self, partial_batch, arg_max: bool = False, mean_of: int = 1, median=False):
        # Get parameters
        _, parameters = self.log_prob(partial_batch, return_parameters=True)

        # Sample from the learned distribution
        x_in, _, x_modality = self.embed(partial_batch)
        output = torch.zeros(x_in.shape[0], device=x_in.device, dtype=x_in.dtype)

        for modality, sample_fn in self.sample_functions(arg_max, mean_of, median).items():
            modality_mask = (x_modality == modality) & torch.isfinite(x_in)
            # noinspection PyTypeChecker
            if torch.any(modality_mask[:, -1]):
                modality_parameters = torch.full(
                    (*x_in.shape, parameters[modality].shape[-1]), float("nan"),
                    device=parameters[modality].device, dtype=parameters[modality].dtype
                )
                modality_parameters[modality_mask] = parameters[modality]
                output[modality_mask[:, -1]] = sample_fn(modality_parameters[modality_mask[:, -1], -1]).to(output)

        return output

    def sample(self, batch: BatchSpec, offset_dim: int = 0, arg_max: bool = False, mean_of: int = 1,
               _overwrite_sampled=False, type_predictor: Callable = None, pbar: bool = False):
        # The first entry in the batch tuple specifies the size of the sampled tensor
        total_dim = batch.data.shape[1]
        x_out = torch.zeros_like(batch.data)
        if type_predictor is not None:
            type_predictors = [type_predictor() for _ in range(batch.data.shape[0])]
        for dim in trange(offset_dim, total_dim, disable=not pbar):
            dim_batch = partial_batch(batch, dim + 1, overwrite_data=None if _overwrite_sampled else x_out)

            # Let the type predictors determine the next type given the previous values
            if type_predictor is not None:
                dim_batch.types[:] = torch.tensor([
                    next(fn) for fn in type_predictors
                ])

            x_out[:, dim] = self._single_dim_sample(dim_batch, arg_max=arg_max, mean_of=mean_of)

            # Tell the type predictors what to expect next
            if type_predictor is not None:
                for idx, fn in enumerate(type_predictors):
                    fn.send(x_out[idx, dim])
        return x_out


class PropertyMTPPFlexTPPModel(FlexTPPModel):
    def __init__(self, embed_event_index=False, **kwargs):
        super().__init__(**kwargs)
        self.embed_event_index = embed_event_index
        if embed_event_index:
            self.event_index_embedding = PositionalEncoding(self.dim_m, self.dropout, lookup_mode=True)

    def embed_position(self, batch):
        position = batch.extras['position_in_event']
        assert torch.all(position < self.dim_embedding.num_embeddings), f"{position} is out of bound"
        embedding = self.dim_embedding(torch.where(
            position == -1, 0, position
        ))
        if self.embed_event_index:
            event_index = batch.extras["event_index"]
            embedding = embedding + self.event_index_embedding(torch.where(
                event_index == -1, 0, event_index
            ))
        return embedding

    def sample_sequence(self, condition, arg_max: bool = False, mean_of: int = 1, median: bool = False,
                        type_position_predictor: Callable = None, max_count: int = None):
        if type_position_predictor is None:
            raise ValueError("type/position predictor must be specified.")
        batch_size = condition.shape[0]
        type_position_predictors = [type_position_predictor() for _ in range(batch_size)]
        types_out = torch.zeros(batch_size, 0, device=condition.device, dtype=torch.long)
        x_out = torch.zeros(batch_size, 0, device=condition.device, dtype=condition.dtype)
        has_ended = torch.zeros(batch_size, device=condition.device, dtype=torch.bool)
        position_in_event = torch.zeros(batch_size, 0, device=condition.device, dtype=torch.long)
        event_index = torch.zeros(batch_size, 0, device=condition.device, dtype=torch.long)
        while True:
            next_types = []
            next_positions = []
            next_event_indices = []
            for item, type_position_predictor in enumerate(type_position_predictors):
                try:
                    next_type, next_position = type_position_predictor.send(
                        x_out[item, -1] if x_out.shape[1] > 0 else None
                    )
                    next_types.append(next_type)
                    next_positions.append(next_position)
                except StopIteration:
                    has_ended[item] = True
                    next_types.append(0)
                    next_positions.append(0)
                if event_index.shape[1] == 0:
                    next_event_indices.append(0)
                else:
                    next_event_indices.append(event_index[item, -1])
                    if next_positions[-1] == 0:
                        next_event_indices[-1] += 1
            if torch.all(has_ended):
                break

            types_out = torch.cat([
                types_out, torch.tensor(next_types, device=condition.device)[:, None]
            ], 1)
            position_in_event = torch.cat([
                position_in_event, torch.tensor(next_positions, device=condition.device)[:, None]
            ], 1)
            event_index = torch.cat([
                event_index, torch.tensor(next_event_indices, device=condition.device)[:, None]
            ], 1)

            batch = BatchSpec(
                torch.cat([
                    x_out,
                    torch.where(
                        has_ended, float('nan'), 0
                    )[:, None],
                ], 1),
                types_out,
                None,
                condition,
                {
                    "position_in_event": position_in_event,
                    "event_index": event_index
                }
            )
            x_out = torch.cat([
                x_out,
                self._single_dim_sample(batch, arg_max=arg_max, mean_of=mean_of, median=median)[:, None]
            ], 1)

            if x_out.shape[1] == max_count:
                break
        return BatchSpec(
            x_out, types_out, None, condition, {
                "position_in_event": position_in_event,
            }
        )
