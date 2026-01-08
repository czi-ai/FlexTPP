from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
from lightning.fabric.utilities import move_data_to_device
from torch.utils.data import Dataset, DataLoader

from flex_tpp.data.base import MODALITY_GAUSSIAN, BaseDataModule, MODALITY_CATEGORICAL, MODALITY_CONTINUOUS, ItemSpec, \
    batch_collate, BatchSpec, log_and_log_abs_det


class NoOverwriteDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise KeyError(f"Key '{key}' already exists.")
        super().__setitem__(key, value)


def deep_compare_dict(metrics, dict1, dict2, prefix=""):
    for key, value1 in dict1.items():
        value2 = dict2[key]
        if isinstance(value1, dict):
            deep_compare_dict(metrics, value1, value2, f"{prefix}{key}/")
        elif isinstance(value1, float):
            metrics[f"{prefix}{key}"].append(value1 - value2)
        elif isinstance(value1, int):
            metrics[f"{prefix}{key}"].append(value1 == value2)
        elif value1 is None:
            continue
        else:
            raise ValueError(f"Comparison not implemented for {type(value1)}")


def deep_log_probs(metrics, values, prefix=""):
    for key, value in values.items():
        if isinstance(value, dict):
            deep_log_probs(metrics, value, f"{prefix}{key}/")
        else:
            metrics[f"{prefix}{key}"].append(value)


class PropertyMTTPDataset(Dataset):
    START_TIME = "S"
    DURATION = "D"
    EVENT_TYPE = "T"
    PROPERTIES = "P"

    DEFAULT_ORDER = "SDTP"

    def __init__(self, time_series, property_types, eos_event_type, overlapping=True,
                 order=DEFAULT_ORDER, conditional=True, gaussian_except_start_time=False):
        self.time_series = time_series
        self.property_types = property_types
        self.eos_event_type = eos_event_type
        self.overlapping = overlapping
        if self.EVENT_TYPE not in order:
            raise ValueError(f"Event type must be modeled, but {order=!r}.")
        if self.PROPERTIES in order and order.index(self.PROPERTIES) < order.index(self.EVENT_TYPE):
            raise ValueError(f"Properties can only be modeled after event type, but {order=!r}.")
        if self.DURATION in order and order.index(self.DURATION) < order.index(self.START_TIME):
            raise ValueError(f"Duration can only be modeled after start time, but {order=!r}.")
        self.order = order
        self.reference_is_start = self.overlapping or self.DURATION not in order
        self.conditional = conditional
        self.gaussian_except_start_time = gaussian_except_start_time
        if gaussian_except_start_time:
            print(property_types)
            self.property_types = {
                k1: {
                    k: v if v != MODALITY_CONTINUOUS else MODALITY_GAUSSIAN
                    for k, v in inner_property_types.items()
                }
                for k1, inner_property_types in property_types.items()
            }
            print(self.property_types)

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        time_series, events = self.time_series[idx]
        event_vector_parts = []
        log_abs_det_vector_parts = []
        type_vector_parts = []
        position_vector_parts = []
        event_index_vector_parts = []
        previous_time = 0.0
        continuous_type = MODALITY_GAUSSIAN if self.gaussian_except_start_time else MODALITY_CONTINUOUS
        for event_idx, (start_time, end_time, event_type, event_props) in enumerate(events):
            assert previous_time <= start_time, "Events are not ordered or violate overlapping constraint"
            start_time_log, start_time_vol_change = log_and_log_abs_det(start_time - previous_time)
            duration_log, duration_vol_change = log_and_log_abs_det(end_time - start_time)
            position = 0
            for entry in self.order:
                if entry == self.START_TIME:
                    data, log_prob_correction, dtypes = [start_time_log], [start_time_vol_change], [MODALITY_CONTINUOUS]
                elif entry == self.DURATION:
                    data, log_prob_correction, dtypes = [duration_log], [duration_vol_change], [continuous_type]
                elif entry == self.EVENT_TYPE:
                    data, log_prob_correction, dtypes = [event_type], [0], [MODALITY_CATEGORICAL]
                elif entry == self.PROPERTIES:
                    data = event_props.values()
                    log_prob_correction = [0] * len(event_props)
                    dtypes = list(map(self.property_types[event_type].get, event_props.keys()))
                else:
                    raise ValueError(f"Unknown event type '{entry}'")
                event_vector_parts += data
                log_abs_det_vector_parts += log_prob_correction
                type_vector_parts += dtypes
                position_vector_parts += range(position, position + len(data))
                event_index_vector_parts += [event_idx] * len(data)
                position += len(data)

            if self.reference_is_start:
                previous_time = start_time
            else:
                previous_time = end_time

        data = torch.tensor(event_vector_parts)
        if data.dtype != torch.float32:
            raise ValueError(data.dtype)
        return ItemSpec(
            data,
            torch.tensor(type_vector_parts),
            torch.tensor(log_abs_det_vector_parts).float(),
            torch.from_numpy(time_series).float() if self.conditional else None,
            {
                "position_in_event": torch.tensor(position_vector_parts),
                "event_index": torch.tensor(event_index_vector_parts),
            }
        )

    def determine_position_and_type(self):
        continuous_type = MODALITY_GAUSSIAN if self.gaussian_except_start_time else MODALITY_CONTINUOUS
        while True:
            for entry_idx, entry in enumerate(self.order):
                if entry == self.START_TIME:
                    start_time = yield MODALITY_CONTINUOUS, entry_idx
                    assert start_time is not None, "Did not receive start time"
                elif entry == self.DURATION:
                    duration = yield continuous_type, entry_idx
                    assert duration is not None, "Did not receive duration"
                elif entry == self.EVENT_TYPE:
                    event_type = yield MODALITY_CATEGORICAL, entry_idx
                    assert event_type is not None, "Did not receive event type"
                    assert event_type == int(event_type), f"Can't parse {event_type} to an event type (int)."
                    if int(event_type) == self.eos_event_type:
                        break
                elif entry == self.PROPERTIES:
                    properties = self.property_types[int(event_type)]
                    for idx, property_type in enumerate(properties.values()):
                        field_value = yield property_type, idx + entry_idx
                        assert field_value is not None, "Did not receive field value"
            # Causes exception in case event_type is referenced before specified
            del event_type

    def parse_event(self, data, types, idx, offset_time, force_event_type=None, log_probs=None):
        # Optional values
        event: List = [None, None, None, None]
        if log_probs is not None:
            event_log_probs: List = [None, None, None, None]
        for entry in self.order:
            if entry == self.START_TIME:
                event[0] = offset_time + data[idx].exp().item()
                if log_probs is not None:
                    event_log_probs[0] = log_probs[idx]
                idx += 1
            elif entry == self.DURATION:
                event[1] = event[0] + data[idx].exp().item()
                if log_probs is not None:
                    event_log_probs[1] = log_probs[idx]
                idx += 1
            elif entry == self.EVENT_TYPE:
                parsed_event_type = int(data[idx].item())
                event[2] = parsed_event_type
                if log_probs is not None:
                    event_log_probs[2] = log_probs[idx]
                idx += 1
            elif entry == self.PROPERTIES:
                event[3] = {}
                if log_probs is not None:
                    event_log_probs[3] = {}
                properties = self.property_types[event[2] if force_event_type is None else force_event_type]
                for key, modality in properties.items():
                    assert properties[key] == types[idx], f"{properties[key]} != {types[idx]} for key {key}"
                    event[3][key] = (
                        data[idx].item()
                        if modality == MODALITY_CONTINUOUS else
                        int(data[idx].item())
                    )
                    if log_probs is not None:
                        event_log_probs[3][key] = log_probs[idx]
                    idx += 1
        if log_probs is not None:
            return tuple(event), idx, tuple(event_log_probs)
        else:
            return tuple(event), idx

    def parse_batch(self, batch: BatchSpec):
        data = []
        batch_data = batch.data.cpu()
        batch_types = batch.types.cpu()
        for item in range(batch_data.shape[0]):
            if batch.condition is not None:
                time_series = batch.condition[item]
            else:
                time_series = None
            idx = 0
            events = []
            offset_time = 0
            while idx < batch_data.shape[1]:
                try:
                    parsed_event, idx = self.parse_event(batch_data[item], batch_types[item], idx, offset_time)
                    events.append(parsed_event)
                    if self.reference_is_start:
                        offset_time = parsed_event[0]
                    else:
                        offset_time = parsed_event[1]
                    if parsed_event[2] == self.eos_event_type:
                        break
                except IndexError as e:
                    print(f"Index error: {e}. Presumably unfinished sequence at batch entry {item}")
                    break
            data.append((time_series, events))
        return data

    @torch.no_grad()
    def validation_metrics(self, batch: BatchSpec, model, argmax=True, mean_of=50, median=False) -> Dict[str, float]:
        log_prob, parameters = model.log_prob(batch, return_parameters=True)
        # Parallel sampling of all next items in vector
        x_out = torch.full_like(batch.data, float("nan"))
        batch_types = batch.types
        for modality, sample_fn in model.sample_functions(argmax, mean_of, median=median).items():
            modality_mask = (batch_types == modality) & torch.isfinite(batch.data)
            # noinspection PyTypeChecker
            if torch.any(modality_mask):
                modality_parameters = torch.full(
                    (*batch.data.shape, parameters[modality].shape[-1]), float("nan"),
                    device=parameters[modality].device, dtype=parameters[modality].dtype
                )
                modality_parameters[modality_mask] = parameters[modality]
                x_out[modality_mask] = sample_fn(modality_parameters[modality_mask]).to(x_out)
        x_out = x_out.cpu()
        batch_types = batch_types.cpu()
        log_prob = log_prob.cpu()

        # Parse prediction according to the schema of the target
        metrics = defaultdict(list)
        log_prob_metrics = defaultdict(list)
        parsed_batch = self.parse_batch(move_data_to_device(batch, "cpu"))
        for item_idx, (time_series, events) in enumerate(parsed_batch):
            idx = 0
            offset_time = 0
            has_seen_eos = False
            for start_time, end_time, event_type, event_props in events:
                assert not has_seen_eos
                true_event = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "event_type": event_type,
                    "event_props": event_props,
                }
                parsed_event = {}
                parsed_log_prob = {}
                (
                    (
                        parsed_event["start_time"],
                        parsed_event["end_time"],
                        parsed_event["event_type"],
                        parsed_event["event_props"]
                    ),
                    idx,
                    (
                        parsed_log_prob["start_time"],
                        parsed_log_prob["end_time"],
                        parsed_log_prob["event_type"],
                        parsed_log_prob["event_props"]
                    )
                ) = self.parse_event(x_out[item_idx], batch_types[item_idx], idx, offset_time,
                                     force_event_type=event_type, log_probs=log_prob[item_idx])
                if self.reference_is_start:
                    offset_time = start_time
                else:
                    offset_time = end_time

                if event_type == self.eos_event_type:
                    metrics["eos"].append(event_type == parsed_event["event_type"])
                    log_prob_metrics["eos"].append(parsed_log_prob["event_type"])
                    has_seen_eos = True
                else:
                    deep_compare_dict(metrics, true_event, parsed_event)
                    deep_log_probs(log_prob_metrics, parsed_log_prob)

        mean_metrics = {}
        all_discrete = []
        all_continuous = []
        for key, values in metrics.items():
            assert all(type(values[0]) == type(value) for value in values), (key, list(map(type, values)))
            if isinstance(values[0], bool):
                mean_metrics[f"error_{key}"] = 1 - np.mean(values)
                all_discrete.extend(values)
            elif isinstance(values[0], float):
                mean_metrics[f"rmse_{key}"] = np.sqrt(np.mean(np.array(values) ** 2))
                all_continuous.extend(values)
            mean_metrics[f"nll_{key}"] = -np.mean(log_prob_metrics[key])
        if "start_time" in log_prob_metrics and "event_type" in log_prob_metrics:
            mean_metrics["nll_start_time_and_type"] = -np.mean(log_prob_metrics["start_time"] + log_prob_metrics["event_type"])
        mean_metrics["rmse"] = np.sqrt(np.mean(np.array(all_continuous) ** 2))
        mean_metrics["error"] = 1 - np.mean(all_discrete)

        return mean_metrics


def parse_event_props(raw_props: List[str]) -> Dict:
    props = {}
    for prop in raw_props:
        key, value = prop.split('=')
        try:
            props[key] = int(value)
        except ValueError:
            try:
                props[key] = float(value)
            except ValueError:
                props[key] = value
    return props


def parse_dataset(file_name, max_num=None):
    if isinstance(file_name, list):
        dataset = file_name
    else:
        dataset = torch.load(file_name, weights_only=False)

    property_types = {}
    discrete_values = defaultdict(set)
    all_timeseries = []
    rng = np.random.default_rng(0)
    eos_event_type = "{EOS}"
    for tspan, data, events in dataset[:max_num]:
        if len(data.shape) == 2:
            data = data.reshape(data.shape[0])

        # Parse events
        parsed_events = []
        for start_time, end_time, event_type, props in events:
            assert 0 <= start_time <= end_time <= tspan.max(), f"{start_time}, {end_time}, {tspan.max()}"
            if event_type not in property_types:
                property_types[event_type] = {}
                for key, value in props.items():
                    assert type(value) in (int, str, np.str_, float), f"Invalid value {value!r}"
                    property_types[event_type][key] = type(value)
            else:
                for key, value_type in property_types[event_type].items():
                    assert key in props, f"Event of type {event_type} does not contain key {key}"
                    assert type(props[key]) == value_type, (
                        f"Inconsistent type for property {key}: {type(props[key])} != {value_type}"
                    )
                for key in props:
                    assert key in property_types[event_type], f"Event has extra key {key}: {list(property_types.keys())}"
            for key, value in props.items():
                if isinstance(value, (str, int)):
                    discrete_values[(event_type, key)].add(value)
            parsed_events.append([start_time, end_time, event_type, props])

        # End of sequence event
        if len(events) != 0:
            eos_start_time = events[-1][1] + rng.beta(2, 2)
        else:
            eos_start_time = 1.0 + rng.beta(2, 2)

        eos_end_time = eos_start_time + rng.beta(2, 2)
        parsed_events.append([eos_start_time, eos_end_time, eos_event_type, {}])
        property_types[eos_event_type] = {}

        all_timeseries.append((data, parsed_events))

    # Map discrete values to integer values
    discrete_value_map = {
        (event_type, prop_name, value): idx
        for (event_type, prop_name), values in discrete_values.items()
        for idx, value in enumerate(sorted(values))
    }
    event_types = sorted(property_types.keys())
    for data, events in all_timeseries:
        for event in events:
            raw_event_type = event[2]
            event[2] = event_types.index(raw_event_type)
            event_props = event[3]
            for key, value in event_props.items():
                if (raw_event_type, key, value) in discrete_value_map:
                    event_props[key] = discrete_value_map[raw_event_type, key, value]
                elif not isinstance(value, float):
                    raise ValueError(f"No discrete lookup for {raw_event_type}/{key}, value is {value}")

    # "dim" = longest event: len([start, end, type, *props])
    dim = 3 + max(len(event_schema) for event_schema in property_types.values())
    max_num_classes = max(
        len(event_types),
        max( [len(unique_values) for unique_values in discrete_values.values()] + [0])
    )
    # Map to data type indicator
    property_types = {
        event_types.index(event_type): {
            key: MODALITY_CONTINUOUS if value is float else MODALITY_CATEGORICAL
            for key, value in event_prop_types.items()
        }
        for event_type, event_prop_types in property_types.items()
    }

    return all_timeseries, dim, max_num_classes, discrete_value_map, property_types, event_types, event_types.index(eos_event_type)


class PropertyMTTPDataModule(BaseDataModule):
    def __init__(self, file_name, max_num=None, conditional=True, order=PropertyMTTPDataset.DEFAULT_ORDER, gaussian_except_start_time=False, **kwargs):
        (
            all_timeseries, dim, max_num_classes, discrete_value_map,
            property_types, event_types, eos_event_type
        ) = parse_dataset(file_name, max_num)
        super().__init__([dim], max_num_classes)

        self.time_series = all_timeseries
        self.kwargs = kwargs
        self.event_types = event_types
        self.discrete_value_map = discrete_value_map

        train_count = int(len(all_timeseries) * 0.8)
        val_count = int(len(all_timeseries) * 0.1)
        test_count = len(all_timeseries) - train_count - val_count
        assert test_count > 0
        self.train_dataset = PropertyMTTPDataset(
            all_timeseries[:train_count],
            property_types,
            eos_event_type,
            conditional=conditional,
            order=order,
            gaussian_except_start_time=gaussian_except_start_time
        )
        self.val_dataset = PropertyMTTPDataset(
            all_timeseries[train_count:train_count + val_count],
            property_types,
            eos_event_type,
            conditional=conditional,
            order=order,
            gaussian_except_start_time=gaussian_except_start_time
        )
        self.test_dataset = PropertyMTTPDataset(
            all_timeseries[-test_count:],
            property_types,
            eos_event_type,
            conditional=conditional,
            order=order,
            gaussian_except_start_time=gaussian_except_start_time
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, collate_fn=batch_collate, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, collate_fn=batch_collate, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, collate_fn=batch_collate, **self.kwargs)
