"""Opt-in loopback RPC test: PETALS_TEST_LOCAL_SWARM=1 pytest tests/test_qwen_swarm.py.

Creates a tiny random checkpoint and a private CPU server. Never joins the public swarm.
"""
import os
import threading
import time

import pytest
import torch
from safetensors.torch import save_file
from test_qwen3_5_moe import random_block, tiny_config

from petals import AutoDistributedModelForCausalLM
from petals.models.qwen3_5_moe.ops import Qwen3_5MoeRMSNorm
from petals.server.server import Server
from petals.utils.convert_block import QuantType


@pytest.mark.skipif(os.getenv("PETALS_TEST_LOCAL_SWARM") != "1", reason="opt-in loopback swarm test")
@pytest.mark.parametrize(
    "ranges, inference_only, automatic",
    [
        (("0:4",), False, False),
        (("0:2", "2:4"), False, False),
        (("0:4",), True, False),
        # Every server asks for 2 layers and lets choose_best_blocks() place them.
        ((2, 2, 2), True, True),
    ],
)
@torch.inference_mode()
def test_private_qwen_swarm(tmp_path, ranges, inference_only, automatic):
    c = tiny_config()
    c.save_pretrained(tmp_path)
    blocks = [random_block(c, i) for i in range(c.num_hidden_layers)]
    embed = torch.randn(c.vocab_size, c.hidden_size) * 0.1
    head = torch.randn_like(embed) * 0.1
    norm = Qwen3_5MoeRMSNorm(c.hidden_size, c.rms_norm_eps)
    state = {
        "model.language_model.embed_tokens.weight": embed,
        "model.language_model.norm.weight": norm.weight,
        "lm_head.weight": head,
    }
    for i, block in enumerate(blocks):
        state.update({f"model.language_model.layers.{i}.{k}": v for k, v in block.state_dict().items()})
    save_file(state, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    servers = []
    errors = []

    def serve(server):
        try:
            server.run()
        except BaseException as e:
            errors.append(e)

    threads = []
    client = None
    try:
        for block_range in ranges:
            peers = [str(a) for a in servers[0].dht.get_visible_maddrs()] if servers else []
            server = Server(
                converted_model_name_or_path=str(tmp_path),
                initial_peers=peers,
                dht_prefix="tiny-qwen",
                throughput=1.0,
                num_blocks=block_range if automatic else None,
                block_indices=None if automatic else block_range,
                # Automatic placement is only meaningful with rebalancing left enabled.
                balance_quality=0.75 if automatic else 0.0,
                mean_block_selection_delay=0.1,
                device="cpu",
                torch_dtype="float32",
                quant_type=QuantType.NONE,
                num_handlers=1,
                max_batch_size=128,
                inference_max_length=32,
                attn_cache_tokens=256,
                max_chunk_size_bytes=4096,
                reachable_via_relay=False,
                use_relay=False,
                use_auto_relay=False,
                host_maddrs=["/ip4/127.0.0.1/tcp/0"],
                cache_dir=str(tmp_path / "cache"),
                update_period=1,
                request_timeout=20,
                step_timeout=20,
                session_timeout=30,
                inference_only=inference_only,
            )
            servers.append(server)
            thread = threading.Thread(target=serve, args=(server,), daemon=True)
            threads.append(thread)
            thread.start()
            deadline = time.monotonic() + 45
            while server.module_container is None or not server.module_container.ready.is_set():
                if errors:
                    raise errors[0]
                if time.monotonic() > deadline:
                    raise TimeoutError("Local Qwen server did not become ready")
                time.sleep(0.1)
        if automatic:
            from petals.data_structures import ServerState
            from petals.utils.dht import compute_spans, get_remote_module_infos

            uids = [f"tiny-qwen.{i}" for i in range(c.num_hidden_layers)]
            infos = get_remote_module_infos(servers[0].dht, uids, latest=True)
            covered = [any(s.state == ServerState.ONLINE for s in info.servers.values()) for info in infos]
            assert all(covered), f"auto placement left gaps: {covered}"
            spans = compute_spans(infos, min_state=ServerState.ONLINE)
            assert len(spans) == len(ranges), "every server should have claimed a range"

        client = AutoDistributedModelForCausalLM.from_pretrained(
            tmp_path,
            initial_peers=[str(a) for a in servers[0].dht.get_visible_maddrs()],
            dht_prefix="tiny-qwen",
            torch_dtype=torch.float32,
            # Petals treats max_retries=0 as "retry forever", so a refused rpc_backward would
            # otherwise hang here instead of surfacing. See docs/qwen3.6-deployment.md.
            max_retries=2 if inference_only else 0,
            request_timeout=20,
            connect_timeout=10,
            use_server_to_server=len(servers) > 1,
        )
        ids = torch.tensor([[1, 7, 3, 6]])

        def local_logits(tokens):
            x = embed[tokens]
            for block in blocks:
                x = block(x)[0]
            return torch.nn.functional.linear(norm(x), head)

        torch.testing.assert_close(client(ids).logits, local_logits(ids), atol=3e-5, rtol=3e-4)
        expected = ids.clone()
        for _ in range(4):
            expected = torch.cat((expected, local_logits(expected)[:, -1].argmax(-1, keepdim=True)), dim=1)
        actual = client.generate(ids, max_new_tokens=4, do_sample=False, eos_token_id=None)
        torch.testing.assert_close(actual, expected)
        with client.inference_session(max_length=16):
            first = client.generate(ids, max_new_tokens=2, do_sample=False, eos_token_id=None)
            last = client.generate(max_new_tokens=2, do_sample=False, eos_token_id=None)
        torch.testing.assert_close(torch.cat((first, last), dim=1), expected)

        # Inference and plain forward keep working above; only training is refused.
        with torch.inference_mode(False):
            hidden = torch.randn(1, 4, c.hidden_size, requires_grad=True)
            outputs = client.transformer.layers(hidden)
            if inference_only:
                with pytest.raises(Exception) as caught:
                    outputs.sum().backward()
                assert "inference_only" in str(caught.value)
                assert hidden.grad is None
            else:
                outputs.sum().backward()
                assert hidden.grad is not None
    finally:
        if client is not None:
            client.transformer.layers.sequence_manager.shutdown()
        for server in servers:
            server.shutdown()
        for thread in threads:
            thread.join(timeout=10)
