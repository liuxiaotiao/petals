import json
import os
import urllib.error
import urllib.request


def probe(url):
    """An HTTP error still proves we reached the host; only a transport failure does not."""
    try:
        urllib.request.urlopen(url, timeout=15)
        return "ok"
    except urllib.error.HTTPError:
        return "ok"
    except Exception:
        return "UNREACHABLE"


def probe_weights(endpoint, model):
    """Pull one real byte of one real shard, end to end, through this endpoint.

    Metadata and file content travel different paths: the API host answers with a redirect
    and the bytes come from a CDN on another domain, and a mirror can do the same. An
    allowlist that knows only the API host therefore lets a name-based check pass while the
    servers retry forever on an empty cache. Fetching actual weight bytes is the only probe
    that distinguishes the two, and it follows whatever endpoint is really in effect instead
    of a hardcoded CDN hostname that a mirror would never use.
    """
    base = "%s/%s/resolve/main" % (endpoint, model)
    try:
        request = urllib.request.Request(base + "/model.safetensors.index.json")
        index = json.loads(urllib.request.urlopen(request, timeout=20).read().decode())
        shard = sorted(set(index["weight_map"].values()))[0]
    except Exception:
        return "NO-INDEX"
    try:
        request = urllib.request.Request(base + "/" + shard, headers={"Range": "bytes=0-0"})
        return "ok" if urllib.request.urlopen(request, timeout=30).read(1) else "EMPTY"
    except Exception:
        return "UNREACHABLE"


free = "n/a"
try:
    import torch

    if not torch.cuda.is_available():
        free = "no-cuda"
    else:
        # Needs a real CUDA context, so this fails when the GPU is already saturated.
        free = "%.1fG" % (torch.cuda.mem_get_info(0)[0] / 1024**3)
except Exception as exc:
    free = "ERR-%s" % type(exc).__name__
endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
endpoint = endpoint.rstrip("/")
model = os.environ.get("QWEN_MODEL") or "Qwen/Qwen3.6-35B-A3B"
# Xet-backed repos serve their bytes from xethub rather than the CDN. With
# HF_HUB_DISABLE_XET set, or through a mirror, that path is never taken, so probing it
# would only report a blocker that is already handled.
if os.environ.get("QWEN_SKIP_XET") or endpoint != "https://huggingface.co":
    xet = "disabled"
else:
    xet = probe("https://cas-server.xethub.hf.co")
print("hub=%s cdn=%s xet=%s vram_free=%s%s" % (
    probe(endpoint + "/api/models"),
    probe_weights(endpoint, model),
    xet,
    free,
    "" if endpoint == "https://huggingface.co" else " via=%s" % endpoint.split("//")[-1],
))
