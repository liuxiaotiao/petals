import fcntl
import gc
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
import torch.mps
from hivemind.utils.logging import get_logger
from transformers import PretrainedConfig

from petals.server.block_utils import get_distinct_block_indices, get_model_block, resolve_block_dtype
from petals.utils.convert_block import QuantType, convert_block
from petals.utils.disk_cache import DEFAULT_CACHE_DIR
from petals.utils.misc import DUMMY_KEY_PAST

logger = get_logger(__name__)

try:
    import speedtest
except ImportError:
    raise ImportError("Please `pip install speedtest-cli==2.1.3`")

if not hasattr(speedtest, "Speedtest"):
    raise ImportError(
        "You are using the wrong speedtest module. Please replace speedtest with speedtest-cli.\n"
        "To do that, run `pip uninstall -y speedtest`. Depending on your python environment, "
        "you may need to run uninstall speedtest two or more times, until it says 'not installed'.\n"
        "After that, please `pip install speedtest-cli==2.1.3` to install the correct version."
    )


def get_server_throughput(
    model_name: str,
    config: PretrainedConfig,
    device: torch.device,
    dtype: Union[str, torch.dtype],
    *,
    num_blocks: int,
    quant_type: QuantType,
    tensor_parallel_devices: Sequence[torch.device],
    reachable_via_relay: bool,
    relay_penalty: float = 0.2,
    force_eval: bool = False,
    cache_dir: Optional[str] = None,
) -> Dict[str, float]:
    dtype = resolve_block_dtype(config, dtype)

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    lock_path = Path(cache_dir, "throughput.lock")
    cache_path = Path(cache_dir, "throughput_v5.json")

    # We use the system-wide lock since only one process at a time can measure the host throughput
    os.makedirs(lock_path.parent, exist_ok=True)
    with open(lock_path, "wb+") as lock_fd:
        logger.info("Loading throughput info")
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        # The OS will release the lock when lock_fd is closed or the process is killed

        cache_key = f"model_{model_name}"
        cache_key += f"_device_{get_device_name(device).replace(' ', '_')}"
        cache_key += f"_dtype_{get_dtype_name(dtype, quant_type)}"
        if len(tensor_parallel_devices) > 1:
            for i, device_i in enumerate(tensor_parallel_devices):
                cache_key += f"_tp{i}_{get_device_name(device_i).replace(' ', '_')}"

        cache = {}
        try:
            if not force_eval and os.path.exists(cache_path):
                with open(cache_path) as cache_fd:
                    cache = json.load(cache_fd)
                assert isinstance(cache, dict)
        except Exception:
            logger.exception(f"Failed to read throughput info from {cache_path}")
            cache = {}

        if cache_key not in cache:
            cache[cache_key] = measure_throughput_info(
                config, device, dtype, quant_type=quant_type, tensor_parallel_devices=tensor_parallel_devices
            )

            try:
                os.makedirs(cache_path.parent, exist_ok=True)
                with open(cache_path, "w") as cache_fd:
                    json.dump(cache, cache_fd)
            except Exception:
                logger.exception(f"Failed to save throughput info in {cache_path}")

    throughput_info = cache[cache_key]

    # Most requests start at some block hosted by a server, then use all next blocks hosted on this server.
    # Assuming the start block index is distributed uniformly, the average number of blocks used per request is
    # E[Uniform{1, 2, ..., num_blocks}] = (num_blocks + 1) / 2
    average_blocks_used = (num_blocks + 1) / 2
    throughput = throughput_info["forward_rps"] / average_blocks_used

    network_rps = throughput_info["network_rps"] * (relay_penalty if reachable_via_relay else 1)
    throughput = min(throughput, network_rps)

    throughput_info["throughput"] = throughput
    logger.info(f"Reporting throughput: {throughput:.1f} tokens/sec for {num_blocks} blocks")

    return throughput_info


def measure_throughput_info(
    config: PretrainedConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    quant_type: QuantType,
    tensor_parallel_devices: Sequence[torch.device],
) -> Dict[str, float]:
    logger.info(
        "Measuring network and compute throughput. This takes about a minute and will be cached for future runs"
    )
    return {
        "inference_rps": measure_compute_rps(
            config,
            device,
            dtype,
            quant_type=quant_type,
            tensor_parallel_devices=tensor_parallel_devices,
            n_tokens=1,
            n_steps=100,
            inference=True,
        ),
        "forward_rps": measure_compute_rps(
            config,
            device,
            dtype,
            quant_type=quant_type,
            tensor_parallel_devices=tensor_parallel_devices,
            n_tokens=1024,
            n_steps=10,
            inference=False,
        ),
        "network_rps": measure_network_rps(config),
    }


def measure_network_rps(
    config: PretrainedConfig, *, timeout: float = 60, default_speed: float = 100e6  # 100 Mbit/s
) -> Optional[float]:
    bits_per_request = config.hidden_size * 16  # Clients usually send 16-bit tensors for forward/backward
    try:
        pipe_recv, pipe_send = mp.Pipe(duplex=False)
        process = mp.Process(target=_measure_bits_per_second, args=(pipe_send,))
        process.start()

        if not pipe_recv.poll(timeout):
            process.terminate()
            raise RuntimeError(f"speedtest did not finish in {timeout} seconds")
        network_info = pipe_recv.recv()
        if "exception" in network_info:
            raise RuntimeError(f"speedtest failed: {network_info['exception']}")

        network_rps = min(network_info["download"], network_info["upload"]) / bits_per_request
        if network_rps == 0:
            raise RuntimeError("speedtest has returned network_rps == 0")

        logger.info(
            f"Network throughput: {network_rps:.1f} tokens/sec "
            f"({network_info['download'] / 1e6:.2f} Mbit/s on download, "
            f"{network_info['upload'] / 1e6:.2f} Mbit/s on upload)"
        )
        return network_rps
    except RuntimeError as e:
        logger.info(f"Network throughput is not available: {e}. Using default of {default_speed / 1e6:.2f} Mbit/s")
        return default_speed / bits_per_request


def _measure_bits_per_second(pipe_send: mp.Pipe):
    try:
        s = speedtest.Speedtest()
        s.get_servers()
        s.get_best_server()
        s.download()
        s.upload()
        pipe_send.send(s.results.dict())
    except Exception as e:
        pipe_send.send({"exception": repr(e)})


def measure_compute_rps(
    config: PretrainedConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    quant_type: QuantType,
    tensor_parallel_devices: Sequence[torch.device],
    n_tokens: int,
    n_steps: int,
    inference: bool,
) -> float:
    device = torch.device(device)
    if not tensor_parallel_devices:
        tensor_parallel_devices = (device,)

    # A hybrid model's variants do not run at the same speed, and the server hosts a mix
    # of them. Advertise the slowest so the swarm does not route work we cannot sustain.
    measured = {
        block_index: _measure_one_block_rps(
            config,
            device,
            dtype,
            quant_type=quant_type,
            tensor_parallel_devices=tensor_parallel_devices,
            n_tokens=n_tokens,
            n_steps=n_steps,
            inference=inference,
            block_index=block_index,
        )
        for block_index in get_distinct_block_indices(config)
    }
    device_rps = min(measured.values())

    devices_repr = get_device_name(device)
    if len(tensor_parallel_devices) > 1:
        device_names = tuple(map(get_device_name, map(torch.device, tensor_parallel_devices)))
        devices_repr = ", ".join(f"{count}x {name}" for name, count in Counter(device_names).most_common())
    variants = ""
    if len(measured) > 1:
        per_block = ", ".join(f"block {index}: {rps:.1f}" for index, rps in sorted(measured.items()))
        variants = f", slowest of {len(measured)} block variants ({per_block})"

    logger.info(
        f"{'Inference' if inference else 'Forward pass'} throughput: {device_rps:.1f} tokens/sec per block "
        f"({n_tokens} tokens/batch, {devices_repr}, {get_dtype_name(dtype, quant_type)}{variants})"
    )
    return device_rps


def _measure_one_block_rps(
    config: PretrainedConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    quant_type: QuantType,
    tensor_parallel_devices: Sequence[torch.device],
    n_tokens: int,
    n_steps: int,
    inference: bool,
    block_index: int,
) -> float:
    with torch.inference_mode():
        block = get_model_block(config, block_index)
        block = block.to(dtype)
        block = convert_block(
            block, block_index, config, tensor_parallel_devices, device, quant_type=quant_type, freeze=True
        )

        cache = (DUMMY_KEY_PAST.to(dtype=dtype, device=device), DUMMY_KEY_PAST.to(dtype=dtype, device=device))
        if getattr(config, "petals_custom_cache", False):
            cache = None
        dummy_input = torch.randn(1, n_tokens, config.hidden_size, device=device, dtype=dtype)

        # Skip the 1st step to exclude the initialization time
        def step(cache_):
            outputs = block.forward(dummy_input, use_cache=inference, layer_past=cache_ if inference else None)
            return outputs[1] if inference else None

        cache = step(cache)
        synchronize(device)

        start_time = time.perf_counter()
        for _ in range(n_steps):
            cache = step(cache)
        synchronize(device)
        elapsed = time.perf_counter() - start_time

    # Blocks are large (MoE experts especially); release one variant before building the next.
    del block, cache, dummy_input, step
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return n_steps * n_tokens / elapsed


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def get_device_name(device: torch.device) -> str:
    return f"{torch.cuda.get_device_name(device)} GPU" if device.type == "cuda" else device.type.upper()


def get_dtype_name(dtype: torch.dtype, quant_type: QuantType) -> str:
    name = str(dtype).replace("torch.", "")
    if quant_type != QuantType.NONE:
        name += f", quantized to {quant_type.name.lower()}"
    return name
