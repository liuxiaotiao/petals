"""Report which layers of a private swarm are online, before you run the client.

A swarm is usable only once every layer has at least one ONLINE server, so this
answers "are the nodes done joining?" without starting a generation that would
otherwise retry until it times out.

  python examples/check_qwen_swarm.py --initial-peers "$BOOTSTRAP_PEER" --watch
"""
import argparse
import sys
import time

from hivemind import DHT

from petals import AutoDistributedConfig
from petals.data_structures import UID_DELIMITER, ServerState
from petals.utils.dht import compute_spans, get_remote_module_infos


def summarize(dht, dht_prefix, num_blocks):
    """Return (per-layer ONLINE counts, ONLINE spans, JOINING spans).

    A server announces JOINING before it has finished loading weights, which for a
    35B model over the Hub can take a long time. Reporting those separately is the
    difference between "still downloading" and "every server died on startup".
    """
    uids = [f"{dht_prefix}{UID_DELIMITER}{index}" for index in range(num_blocks)]
    infos = get_remote_module_infos(dht, uids, latest=True)
    online = [sum(server.state == ServerState.ONLINE for server in info.servers.values()) for info in infos]
    ready = compute_spans(infos, min_state=ServerState.ONLINE)
    joining = {
        peer: span for peer, span in compute_spans(infos, min_state=ServerState.JOINING).items() if peer not in ready
    }
    return online, ready, joining


def format_ranges(indices):
    """Collapse [22, 23, 24, 27] into '22:25, 27:28' so long gaps stay readable."""
    ranges, start = [], None
    for index in indices + [None]:
        if start is None:
            start, previous = index, index
        elif index != previous + 1:
            ranges.append(f"{start}:{previous + 1}")
            start, previous = index, index
        else:
            previous = index
        if index is None:
            break
    return ", ".join(ranges)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-peers", nargs="+", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--revision", default=None)
    # Given both of these, the check runs without reading the model config at all.
    parser.add_argument("--dht-prefix", default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--watch", action="store_true", help="Poll until every layer is online")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    if args.dht_prefix and args.num_blocks:
        dht_prefix, num_blocks = args.dht_prefix, args.num_blocks
    else:
        config = AutoDistributedConfig.from_pretrained(args.model, revision=args.revision, dht_prefix=args.dht_prefix)
        dht_prefix, num_blocks = config.dht_prefix, config.num_hidden_layers

    dht = DHT(initial_peers=args.initial_peers, client_mode=True, start=True)
    try:
        deadline = time.monotonic() + args.timeout
        while True:
            online, spans, joining = summarize(dht, dht_prefix, num_blocks)
            missing = [index for index, count in enumerate(online) if count == 0]

            print(
                f"\nDHT prefix {dht_prefix}, {num_blocks} layers, "
                f"{len(spans)} server(s) online, {len(joining)} still joining"
            )
            for peer_id, span in sorted(spans.items(), key=lambda item: item[1].start):
                info = span.server_info
                rps = f"{info.inference_rps:.1f}" if info.inference_rps else "n/a"
                print(
                    f"  …{str(peer_id)[-6:]}  layers {span.start}:{span.end}"
                    f"  {info.torch_dtype}/{info.quant_type}"
                    f"  inference_rps={rps}  cache_tokens_left={info.cache_tokens_left}"
                )
            for peer_id, span in sorted(joining.items(), key=lambda item: item[1].start):
                print(f"  …{str(peer_id)[-6:]}  layers {span.start}:{span.end}  JOINING (loading weights)")
            if not missing:
                weakest = min(online)
                print(f"Every layer is online (thinnest layer has {weakest} server(s)). The swarm is usable.")
                return 0
            print(f"Missing layers: {format_ranges(missing)} — the client cannot generate yet.")
            if not spans and not joining:
                print(
                    "No server has announced anything under this prefix. Either none of them started,\n"
                    "or they are using a different --dht_prefix. Check a server log:\n"
                    "  bash examples/qwen_cluster.sh logs <node-id> 60"
                )

            if not args.watch or time.monotonic() > deadline:
                return 1
            time.sleep(args.interval)
    finally:
        dht.shutdown()


if __name__ == "__main__":
    sys.exit(main())
