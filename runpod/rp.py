#!/usr/bin/env python3
"""rp.py - local RunPod helper for the ATCNet study (runs on the workstation, not on the pod).

API surfaces (field names verified against the live API and its OpenAPI documents):
  REST v2   https://api.runpod.io/v2      pods, catalog, ssh-keys, network volumes (current)
  REST v1   https://rest.runpod.io/v1     deprecated (retires 2026-11-15); used only for COMMUNITY
                                          pods (v2 has no supportPublicIp) and --interruptible
  GraphQL   https://api.runpod.io/graphql balance and per-cloud lowest price with vCPU/RAM filters.
                                          Also scheduled for retirement; v2 /billing and
                                          /catalog/gpus can replace it.
The API key is read at run time from $RUNPOD_API_KEY or ~/.runpod/api_key. It is never printed,
logged or written anywhere; error bodies are scrubbed before they are shown.

Safety model
  * balance, gpus, status, ssh-cmd, pull are read-only (GET requests and GraphQL queries).
  * create, start, stop, terminate are DRY RUNS unless --yes is given: they print the exact
    request (key redacted) and send nothing.
  * bundle-data, bundle-code, verify-results work on local files only (no network).

Examples
  python rp.py balance
  python rp.py gpus --available --cloud SECURE --min-vcpu 12 --min-ram 32
  python rp.py create --gpu "NVIDIA RTX A4000" --min-vcpu 12 --name atcnet        # dry run
  python rp.py status <pod_id> --wait 900
  python rp.py ssh-cmd <pod_id> --local-results ./pod_results
  python rp.py bundle-data --data-root "$EEGDATA" --out - | ssh ... "tar xzf - -C /workspace/data"
  python rp.py verify-results ./pod_results/results_bonn_atcnet_<stamp>.tgz
  python rp.py stop <pod_id> --yes
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import sys
import tarfile
import time
from pathlib import Path

GRAPHQL_URL = "https://api.runpod.io/graphql"
V2 = "https://api.runpod.io/v2"
V1 = "https://rest.runpod.io/v1"
KEY_FILE = Path.home() / ".runpod" / "api_key"
DEFAULT_PUBKEY = Path.home() / ".ssh" / "id_ed25519.pub"
DEFAULT_PRIVKEY = "~/.ssh/id_ed25519"
# Image of RunPod's own "PyTorch 2.8.0" template (most likely cached on hosts). Keep ONE tag for the
# whole study: package versions are part of the notebooks' STUDY_ID.
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
MIN_CUDA = "12.8"                      # the image is built on CUDA 12.8.1
CANDIDATES = [
    "NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX A5000", "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090",
    "NVIDIA L4", "NVIDIA A40", "NVIDIA L40S", "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA RTX 4000 Ada Generation", "NVIDIA RTX 4000 SFF Ada Generation",
]
POD_WS = "/workspace"
POD_REPO = f"{POD_WS}/code/seizure_study"
POD_RS = f"{POD_REPO}/runpod/remote_setup.sh"
TIMEOUT = 60
REDACTED = "***REDACTED***"


class ApiError(Exception):
    """An API call failed; the message never contains the key."""


# --------------------------------------------------------------------------- key + HTTP

_KEY_CACHE: dict = {}


def _key() -> str:
    if "k" not in _KEY_CACHE:
        k = os.environ.get("RUNPOD_API_KEY", "").strip()
        if not k:
            try:
                k = KEY_FILE.read_text(encoding="utf-8").strip()
            except OSError:
                raise ApiError(f"no API key: set RUNPOD_API_KEY or create {KEY_FILE}") from None
        if not k:
            raise ApiError(f"{KEY_FILE} is empty")
        _KEY_CACHE["k"] = k
    return _KEY_CACHE["k"]


def _scrub(text: str) -> str:
    k = _KEY_CACHE.get("k") or os.environ.get("RUNPOD_API_KEY", "").strip()
    return text.replace(k, REDACTED) if k else text


def _headers() -> dict:
    return {"Authorization": "Bearer " + _key(), "Content-Type": "application/json",
            "User-Agent": "rp.py/2.0"}


def _redacted_headers() -> dict:
    return {"Authorization": f"Bearer {REDACTED} (read from ~/.runpod/api_key at send time)",
            "Content-Type": "application/json"}


def http(method: str, url: str, *, params=None, body=None, ok=(200, 201, 204), retries=2):
    """One HTTP call; GETs are retried on 429/5xx and network errors. Returns parsed JSON or None."""
    import requests
    for attempt in range(retries + 1):
        try:
            r = requests.request(method, url, params=params, json=body, headers=_headers(), timeout=TIMEOUT)
        except requests.RequestException as e:          # message holds the URL only, never headers
            if attempt < retries and method == "GET":
                time.sleep(2 * (attempt + 1))
                continue
            raise ApiError(f"{method} {url} failed: {_scrub(str(e))}") from None
        if r.status_code in (429, 500, 502, 503, 504) and method == "GET" and attempt < retries:
            time.sleep(float(r.headers.get("Retry-After", 2 * (attempt + 1))))
            continue
        if r.status_code not in ok:
            raise ApiError(f"{method} {url} -> HTTP {r.status_code}: {_scrub(r.text[:1500])}")
        if r.status_code == 204 or not r.content:
            return None
        try:
            return r.json()
        except ValueError:
            return r.text
    return None


def gql(query: str, variables: dict | None = None) -> dict:
    j = http("POST", GRAPHQL_URL, body={"query": query, "variables": variables or {}}, ok=(200,))
    if isinstance(j, dict) and j.get("errors"):
        raise ApiError("GraphQL: " + _scrub(json.dumps(j["errors"])[:1500]))
    return (j or {}).get("data") or {}


def print_request(method: str, url: str, body=None, api_note: str = ""):
    print("DRY RUN - nothing was sent. Re-run with --yes to send exactly this request:")
    if api_note:
        print("  API:", api_note)
    print(f"  {method} {url}")
    for k, v in _redacted_headers().items():
        print(f"  {k}: {v}")
    if body is not None:
        print("  Body:")
        print("\n".join("    " + line for line in json.dumps(body, indent=2).splitlines()))


def _fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(rows, cols):
    widths = {c: max([len(c)] + [len(_fmt(r.get(c))) for r in rows]) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def list_pods() -> list:
    return (http("GET", f"{V2}/pods") or {}).get("pods", [])


# --------------------------------------------------------------------------- balance

def cmd_balance(a):
    me = gql("query { myself { clientBalance currentSpendPerHr spendLimit underBalance minBalance } }")["myself"]
    pods = list_pods()
    vols = http("GET", f"{V2}/network-volumes") or {}
    keys = (http("GET", f"{V2}/account/ssh-keys") or {}).get("keys", [])
    try:
        local = DEFAULT_PUBKEY.read_text(encoding="utf-8").split()[1]
    except (OSError, IndexError):
        local = ""
    registered = bool(local) and any(local in k for k in keys)
    bal = float(me.get("clientBalance") or 0.0)
    if a.json:
        print(json.dumps(dict(myself=me, pods=len(pods), network_volumes=len(vols.get("networkVolumes", [])),
                              registered_ssh_keys=len(keys), local_id_ed25519_registered=registered), indent=2))
        return
    print(f"Balance (USD):            {bal:.4f}")
    print(f"Current spend (USD/h):    {_fmt(me.get('currentSpendPerHr'))}")
    print(f"Spend limit (USD/h):      {_fmt(me.get('spendLimit'))}")
    print(f"Under balance flag:       {me.get('underBalance')}")
    print(f"Pods:                     {len(pods)}")
    print(f"Network volumes:          {len(vols.get('networkVolumes', []))}")
    print(f"Registered SSH keys:      {len(keys)}  (local {DEFAULT_PUBKEY.name} registered: {registered})")
    spend = float(me.get("currentSpendPerHr") or 0)
    if spend > 0:
        print(f"Balance lasts about {bal / spend:.1f} h at the current spend.")
    if bal < 1.0:
        print("\nNOTE: an on-demand pod needs at least one hour of credit to deploy, and at $0 a pod")
        print("without a network volume is terminated (its /workspace is lost). Top up first.")


# --------------------------------------------------------------------------- gpus

def _lowest_price_query(min_vcpu, min_ram) -> str:
    extra = ""
    if min_vcpu:
        extra += f", minVcpuCount:{int(min_vcpu)}"
    if min_ram:
        extra += f", minMemoryInGb:{int(min_ram)}"
    fields = "uninterruptablePrice minimumBidPrice minVcpu minMemory stockStatus"
    return ("query { gpuTypes { id displayName memoryInGb securePrice communityPrice "
            f"sec: lowestPrice(input:{{gpuCount:1, secureCloud:true{extra}}}) {{ {fields} }} "
            f"com: lowestPrice(input:{{gpuCount:1, secureCloud:false{extra}}}) {{ {fields} }} }} }}")


def gpu_inventory(min_vcpu=None, min_ram=None) -> dict:
    """v2 catalog availability per cloud, merged with GraphQL lowest price under the vCPU/RAM filter."""
    cat: dict = {}
    for cloud in ("SECURE", "COMMUNITY"):
        j = http("GET", f"{V2}/catalog/gpus",
                 params={"include": "AVAILABILITY", "product": "POD", "cloud": cloud, "count": 1}) or {}
        for g in j.get("gpus", []):
            e = cat.setdefault(g["id"], dict(id=g["id"], name=g.get("name"), vram=g.get("memory"),
                                             price=g.get("price", {}),
                                             offered=dict(SECURE=g.get("secure"), COMMUNITY=g.get("community"))))
            e[cloud] = dict(availability=g.get("availability"),
                            dcs=[d["id"] for d in g.get("dataCenters", []) if d.get("availability") not in (None, "NONE")],
                            cuda=[c["version"] for c in g.get("cudaVersions", []) if c.get("available")])
    data = gql(_lowest_price_query(min_vcpu, min_ram))
    for g in data.get("gpuTypes", []):
        e = cat.setdefault(g["id"], dict(id=g["id"], name=g.get("displayName"), vram=g.get("memoryInGb"),
                                         price={}, offered={}))
        e["gql"] = dict(SECURE=g.get("sec") or {}, COMMUNITY=g.get("com") or {})
    return cat


def _cuda_ok(versions) -> bool:
    def key(v):
        try:
            return tuple(int(x) for x in str(v).split("."))
        except ValueError:
            return (0,)
    return any(key(v) >= key(MIN_CUDA) for v in versions)


def gpu_rows(cat: dict, *, clouds, include_all=False, available=False, min_vcpu=None, min_ram=None) -> list:
    """Table rows. With a vCPU/RAM filter a row matches only if GraphQL reports stock for a machine
    meeting that filter; the v2 catalog's availability ignores the filter and is shown for context."""
    filtered = bool(min_vcpu or min_ram)
    ids = sorted(cat) if include_all else [c for c in CANDIDATES if c in cat]
    rows = []
    for gid in ids:
        e = cat[gid]
        for cloud in clouds:
            if (e.get("offered") or {}).get(cloud) is False and not include_all:
                continue
            v2 = e.get(cloud) or {}
            lp = (e.get("gql") or {}).get(cloud) or {}
            avail = v2.get("availability") or "NONE"
            stock = lp.get("stockStatus")
            if filtered:
                match = bool(stock)
                state = "ok" if match else ("filter unmet" if avail != "NONE" else "no stock")
            else:
                match = avail != "NONE" or bool(stock)
                state = "ok" if match else "no stock"
            if available and not match:
                continue
            live = lp.get("uninterruptablePrice") if stock or not filtered else None
            vcpu = lp.get("minVcpu") if stock else None
            cuda = v2.get("cuda", [])
            rows.append(dict(gpu=gid.replace("NVIDIA ", "").replace("GeForce ", ""), gpu_id=gid, vram=e.get("vram"),
                             cloud=cloud[:3], list_usd_h=(e.get("price") or {}).get(cloud.lower()), live_usd_h=live,
                             avail_v2=avail, stock=stock, vcpu=vcpu, ram_gb=lp.get("minMemory") if stock else None,
                             match=state, cuda_max=(max(cuda, key=lambda v: tuple(int(x) for x in v.split(".")))
                                                    if cuda else None),
                             cuda_ok=(_cuda_ok(cuda) if cuda else None),
                             dcs=",".join(v2.get("dcs", [])[:3]) or None))
    if available:
        rows.sort(key=lambda r: (r["live_usd_h"] if r["live_usd_h"] is not None else r["list_usd_h"] or 99))
    return rows


def cmd_gpus(a):
    cat = gpu_inventory(a.min_vcpu, a.min_ram)
    clouds = ["SECURE", "COMMUNITY"] if a.cloud == "BOTH" else [a.cloud]
    rows = gpu_rows(cat, clouds=clouds, include_all=a.all, available=a.available,
                    min_vcpu=a.min_vcpu, min_ram=a.min_ram)
    if a.json:
        print(json.dumps(rows if not a.raw else cat, indent=2))
        return
    parts = ([f">= {a.min_vcpu} vCPU"] if a.min_vcpu else []) + ([f">= {a.min_ram} GB RAM"] if a.min_ram else [])
    filt = f", placement filter {', '.join(parts)}" if parts else ""
    print(f"RunPod GPU pods, 1 GPU{filt}. USD/h per pod. vcpu/ram = smallest machine in stock at that price.")
    print("match: ok = in stock for this filter; 'filter unmet' = GPU in stock, but not on a machine with")
    print("these vCPU/RAM (create would fail); stock changes minute to minute.\n")
    _table(rows, ["gpu", "vram", "cloud", "list_usd_h", "live_usd_h", "avail_v2", "stock", "vcpu", "ram_gb",
                  "match", "cuda_max", "dcs"])
    print(f"\nThe default image needs host CUDA >= {MIN_CUDA} (create sets this filter). "
          "Spot prices currently equal on-demand prices.")


# --------------------------------------------------------------------------- create

def _read_pubkey(path: Path) -> str:
    try:
        txt = path.expanduser().read_text(encoding="utf-8").strip()
    except OSError as e:
        raise ApiError(f"cannot read public key {path}: {e}") from None
    if not txt.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")) or "PRIVATE KEY" in txt:
        raise ApiError(f"{path} does not look like an OpenSSH PUBLIC key")
    return txt.splitlines()[0]


def build_create_request(a):
    """Returns (method, url, body, note). No network access."""
    pub = _read_pubkey(Path(a.pubkey))
    ports = ["22/tcp"] + (["8888/http"] if a.jupyter else [])
    api = a.api
    if api == "auto":
        api = "v1" if (a.cloud == "COMMUNITY" or a.interruptible) else "v2"
    if api == "v2":
        if a.interruptible:
            raise ApiError("REST v2 has no interruptible flag; use --api v1 (deprecated) or drop --interruptible")
        body = {
            "name": a.name,
            "image": a.image,
            "cloud": a.cloud,
            "gpu": {"id": a.gpu, "count": a.gpu_count, "minVcpuCountPerGpu": a.min_vcpu,
                    "minRamPerGpu": a.min_ram, "minCudaVersion": MIN_CUDA},
            "disk": a.container_disk,
            "ports": ports,
            "env": {"PUBLIC_KEY": pub},
            "startSsh": True,
            "startJupyter": bool(a.jupyter),
        }
        if a.network_volume:
            body["mounts"] = {"network": [{"volumeId": a.network_volume, "path": POD_WS}]}
        elif a.volume_disk:
            body["mounts"] = {"persistent": {"size": a.volume_disk, "path": POD_WS}}
        if a.datacenter:
            body["dataCenterIds"] = a.datacenter
        return "POST", f"{V2}/pods", body, "REST v2 (current)"
    # v1: no allowedCudaVersions filter. Its enum stops at 13.0 and would exclude the newest hosts;
    # every host driver >= 12.8 runs the image, and `remote_setup.sh setup` checks the driver.
    body = {
        "name": a.name,
        "imageName": a.image,
        "cloudType": a.cloud,
        "computeType": "GPU",
        "gpuTypeIds": [a.gpu],
        "gpuCount": a.gpu_count,
        "minVCPUPerGPU": a.min_vcpu,
        "minRAMPerGPU": a.min_ram,
        "containerDiskInGb": a.container_disk,
        "volumeInGb": 0 if a.network_volume else a.volume_disk,
        "volumeMountPath": POD_WS,
        "ports": ports,
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": pub},
        "interruptible": bool(a.interruptible),
    }
    if a.network_volume:
        body["networkVolumeId"] = a.network_volume
    if a.datacenter:
        body["dataCenterIds"] = a.datacenter
    return "POST", f"{V1}/pods", body, "REST v1 (deprecated, retires 2026-11-15; used for supportPublicIp/interruptible)"


def _same_name_pods(name: str) -> list:
    return [p for p in list_pods() if p.get("name") == name and p.get("status") not in ("TERMINATED",)]


def _warn_possible_pod(name: str):
    """After a failed or timed-out create: the request may still have created a pod."""
    print("\nWARNING: the create request failed or timed out, but RunPod may still have created the pod.",
          file=sys.stderr)
    try:
        same = _same_name_pods(name)
    except ApiError as e:
        print(f"  (could not list pods: {e})", file=sys.stderr)
        same = None
    if same:
        for p in same:
            print(f"  FOUND pod '{name}': id={p.get('id')} status={p.get('status')} cost/h={p.get('cost')}",
                  file=sys.stderr)
        print("  It is billing. Use it (rp.py status <id>) or terminate it; do NOT re-run create.", file=sys.stderr)
    else:
        print("  Run `python rp.py status` before retrying: a duplicate pod would bill in parallel.", file=sys.stderr)


def cmd_create(a):
    method, url, body, note = build_create_request(a)
    if a.cloud == "COMMUNITY" and url.startswith(V2):
        print("WARNING: v2 cannot request a public IP; a COMMUNITY host without one gives proxy-only SSH "
              "(no scp/tar). Prefer --api v1 for COMMUNITY.")
    if url.startswith(V1):
        print("NOTE: v1 create sends no CUDA filter; setup/preflight on the pod checks the driver (>= 12.8).")
    # Read-only pre-flight: duplicates, stock, price, balance. Advisory in dry-run mode.
    try:
        same = _same_name_pods(a.name)
        if same:
            ids = ", ".join(f"{p.get('id')} ({p.get('status')})" for p in same)
            msg = f"a pod named '{a.name}' already exists: {ids}"
            if a.yes and not a.allow_duplicate:
                raise ApiError(msg + ". Refusing to create a second one (use --allow-duplicate or another --name).")
            print("WARNING:", msg)
        cat = gpu_inventory(a.min_vcpu, a.min_ram)
        e = cat.get(a.gpu)
        if e is None:
            print(f"WARNING: GPU id '{a.gpu}' is not in the live catalog (see `rp.py gpus --all`).")
        else:
            lp = (e.get("gql") or {}).get(a.cloud) or {}
            av = (e.get(a.cloud) or {}).get("availability")
            price = lp.get("uninterruptablePrice") or (e.get("price") or {}).get(a.cloud.lower())
            print(f"Pre-flight: {a.gpu} / {a.cloud}: v2 availability={av}, stock for >= {a.min_vcpu} vCPU / "
                  f">= {a.min_ram} GB = {lp.get('stockStatus')}, ~${_fmt(price)}/h per GPU")
            if not lp.get("stockStatus"):
                print("WARNING: no machine in stock matches this GPU/cloud/vCPU/RAM; create would likely fail.")
            me = gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
            bal, spend = float(me.get("clientBalance") or 0), float(me.get("currentSpendPerHr") or 0)
            print(f"Pre-flight: balance ${bal:.2f}, already spending ${spend:.2f}/h; "
                  f"volume disk while stopped ~${(a.volume_disk or 0) * 0.20 / 730:.4f}/h")
            if price and bal < float(price) * a.gpu_count:
                print("WARNING: balance is below one hour of this pod; RunPod refuses to deploy (HTTP 402).")
            elif price:
                print(f"Pre-flight: the balance covers about {bal / (float(price) * a.gpu_count + spend):.1f} h "
                      "with this pod added to what is already running.")
    except ApiError as ex:
        if a.yes and "Refusing" in str(ex):
            raise
        print(f"(pre-flight incomplete: {ex})")
    if not a.yes:
        print_request(method, url, body, note)
        return
    print(f"Sending {method} {url} ...")
    try:
        pod = http(method, url, body=body, ok=(200, 201), retries=0)
    except (ApiError, KeyboardInterrupt):
        _warn_possible_pod(a.name)
        raise
    pod = pod or {}
    print(json.dumps({k: pod.get(k) for k in ("id", "name", "status", "desiredStatus", "costPerHr", "cost",
                                              "gpu", "machine", "publicIp", "portMappings") if pod.get(k) is not None},
                     indent=2))
    if pod.get("id"):
        print(f"\nCreated pod {pod['id']}. Next: python rp.py status {pod['id']} --wait 900 ; "
              f"python rp.py ssh-cmd {pod['id']}")
    else:
        _warn_possible_pod(a.name)


# --------------------------------------------------------------------------- status / ssh

def _get_pod(pid: str) -> dict:
    return http("GET", f"{V2}/pods/{pid}") or {}


def _load_pod(a) -> dict:
    if getattr(a, "from_json", None):         # offline rendering from a saved `status <id> --json`
        return json.loads(Path(a.from_json).read_text(encoding="utf-8"))
    return _get_pod(a.pod_id)


def _direct_ssh(p: dict):
    d = (p.get("ssh") or {}).get("direct") or None
    if d and d.get("host") and d.get("port"):
        return d["host"], int(d["port"])
    for port in ((p.get("runtime") or {}).get("ports") or []):
        if port.get("private") == 22 and port.get("public") and port.get("ip"):
            return port["ip"], int(port["public"])
    return None


def _pod_row(p: dict) -> dict:
    g = p.get("gpu") or {}
    rt = p.get("runtime") or {}
    gpus = rt.get("gpus") or []
    d = _direct_ssh(p)
    return dict(id=p.get("id"), name=p.get("name"), status=p.get("status"),
                gpu=(f"{g.get('count', 1)}x {g.get('id', '')}".replace("NVIDIA ", "") if g else "-"),
                vcpu=g.get("vcpuCount"), ram_gb=g.get("memory"), cloud=p.get("cloud"), dc=p.get("dataCenterId"),
                usd_h=p.get("cost"), uptime_min=(round(rt["uptime"] / 60) if rt.get("uptime") else None),
                gpu_util=(",".join(str(x.get("util")) for x in gpus) if gpus else None),
                cpu_util=(rt.get("cpu") or {}).get("util"), mem_util=(rt.get("memory") or {}).get("util"),
                ssh=(f"{d[0]}:{d[1]}" if d else ("proxy-only" if (p.get("ssh") or {}).get("proxy") else None)))


def cmd_status(a):
    if not a.pod_id:
        pods = list_pods()
        if a.json:
            print(json.dumps(pods, indent=2))
            return
        if not pods:
            print("No pods.")
            return
        _table([_pod_row(p) for p in pods], ["id", "name", "status", "gpu", "vcpu", "ram_gb", "cloud", "dc",
                                              "usd_h", "uptime_min", "gpu_util", "cpu_util", "ssh"])
        running = sum(float(p.get("cost") or 0) for p in pods if p.get("status") == "RUNNING")
        if running:
            print(f"\nRunning pods bill about ${running:.2f}/h in total.")
        return
    deadline = time.time() + (a.wait or 0)
    while True:
        p = _get_pod(a.pod_id)
        ready = p.get("status") == "RUNNING" and _direct_ssh(p) is not None
        if a.json:
            print(json.dumps(p, indent=2))
        else:
            print(time.strftime("%H:%M:%S"), "  ".join(f"{k}={_fmt(v)}" for k, v in _pod_row(p).items()))
        if not a.wait or ready or time.time() > deadline or p.get("status") in ("ERROR", "TERMINATED", "EXITED"):
            break
        time.sleep(15)
    if not a.json:
        print("allowed actions:", p.get("actions"), "| image:", p.get("image"), "| mounts:", json.dumps(p.get("mounts")))


def _ssh_parts(p: dict, keyfile: str):
    d = _direct_ssh(p)
    if not d:
        return None
    host, port = d
    o = f"-i {keyfile} -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
    return host, f"ssh {o} -p {port} root@{host}", f"scp {o} -P {port}"


def _no_direct_ssh(p: dict, keyfile: str):
    proxy = (p.get("ssh") or {}).get("proxy") or {}
    print("# No direct (public IP) SSH mapping for 22/tcp: pod starting, stopped, or host without public IP.")
    if proxy.get("command"):
        print("# Proxy SSH (interactive shell ONLY; no scp/sftp/rsync/pipes):")
        print(f"{proxy['command']} -i {keyfile}")
    print("# File fallback without public IP: runpodctl send <file> (pod) / runpodctl receive <code> (local).")


def cmd_ssh(a):
    p = _load_pod(a)
    print(f"# pod {p.get('id')} ({p.get('name')}) status={p.get('status')} cloud={p.get('cloud')} "
          f"dc={p.get('dataCenterId')}")
    parts = _ssh_parts(p, a.key)
    if not parts:
        _no_direct_ssh(p, a.key)
        return
    host, ssh, scp = parts
    repo, lr, q = a.repo, a.local_results, a.queue
    print("\n# --- interactive shell (PowerShell or Git Bash)")
    print(ssh)
    print("\n# --- 1. create the folders on the pod first (scp does not create missing parents)")
    print(f"{ssh} \"mkdir -p {POD_WS}/code {POD_WS}/data {POD_WS}/packs\"")
    print("\n# --- 2. upload the code: the repository folder (runpod/, notebooks/, protocol/, atcnet/, chbmit/)")
    print("#     Git Bash, one tar stream (excludes .git, caches, checkpoints):")
    print(f"python \"{repo}/runpod/rp.py\" bundle-code --repo \"{repo}\" --out - | {ssh} \"tar xzf - -C {POD_WS}/code\"")
    print("#     or PowerShell / Git Bash with scp (copies the folder as /workspace/code/seizure_study):")
    print(f"{scp} -r \"{repo}\" root@{host}:{POD_WS}/code/")
    print("\n# --- 3. upload the data (Git Bash: streamed, no local archive)")
    print(f"python \"{repo}/runpod/rp.py\" bundle-data --data-root \"{a.local_data}\" --out - | "
          f"{ssh} \"tar xzf - -C {POD_WS}/data\"")
    print("#     or build the archive once and scp it (PowerShell-safe):")
    print(f"python \"{repo}/runpod/rp.py\" bundle-data --data-root \"{a.local_data}\" --out data_bundle.tgz")
    print(f"{scp} data_bundle.tgz root@{host}:{POD_WS}/")
    print(f"{ssh} \"tar xzf {POD_WS}/data_bundle.tgz -C {POD_WS}/data && rm {POD_WS}/data_bundle.tgz\"")
    print("\n# --- 4. set up and check the pod")
    print(f"{ssh} \"bash {POD_RS} setup && bash {POD_RS} check-data && bash {POD_RS} preflight\"")
    print(f"{ssh} \"bash {POD_RS} progress {q}\"")
    print("\n# --- 5. download results (then verify locally)")
    _print_pull(ssh, scp, host, q, lr, repo)
    print("\n# --- optional Jupyter tunnel (pod created with --jupyter): then open http://localhost:8888")
    print(ssh.replace("ssh ", "ssh -N -L 8888:localhost:8888 ", 1))


def _print_pull(ssh, scp, host, q, lr, repo):
    print(f"{ssh} \"bash {POD_RS} pack {q}\"")
    print(f"mkdir -p \"{lr}\"")
    print(f"{scp} \"root@{host}:{POD_WS}/packs/results_{q}_*\" \"{lr}/\"")
    print(f"python \"{repo}/runpod/rp.py\" verify-results \"{lr}\"/results_{q}_<stamp>.tgz")


def cmd_pull(a):
    p = _load_pod(a)
    parts = _ssh_parts(p, a.key)
    if not parts:
        _no_direct_ssh(p, a.key)
        return
    host, ssh, scp = parts
    print(f"# pod {p.get('id')} ({p.get('name')}) status={p.get('status')}")
    print("# 1. pack on the pod (safe while workers run) 2. copy tarball + .sha256 + .manifest.sha256 "
          "3. verify locally")
    _print_pull(ssh, scp, host, a.queue, a.local_results, a.repo)
    print("# Git Bash alternative without a pod-side archive (no integrity sidecar):")
    print(f"{ssh} \"cd {POD_WS} && tar czf - results/{a.queue} logs/{a.queue} jobs/{a.queue}\" > "
          f"\"{a.local_results}/stream_{a.queue}.tgz\"")


# --------------------------------------------------------------------------- lifecycle (guarded)

def _lifecycle(a, action: str):
    if action == "terminate":
        method, url, body = "DELETE", f"{V2}/pods/{a.pod_id}", None
    else:
        method, url, body = "POST", f"{V2}/pods/{a.pod_id}/action", {"action": action}
    try:
        p = _get_pod(a.pod_id)
        print(f"Pod {a.pod_id}: status={p.get('status')} allowed={p.get('actions')} "
              f"mounts={json.dumps(p.get('mounts'))} cost/h={p.get('cost')}")
    except ApiError as ex:
        if a.yes:
            raise
        print(f"(could not read pod {a.pod_id}: {ex})")
    if action == "terminate":
        print("TERMINATE permanently deletes the pod and its /workspace volume disk (a network volume is only "
              "detached). Download and verify results first.")
    elif action == "stop":
        print("STOP releases the GPU and wipes the container disk; /workspace is kept (~$0.20/GB/month).")
        print("A restart may find no free GPU on that host (then start without GPU to copy data, or terminate).")
    if not a.yes:
        print_request(method, url, body, "REST v2")
        return
    r = http(method, url, body=body, ok=(200, 201, 204), retries=0)
    print(f"{action}: sent.", ("new status: " + str((r or {}).get("status"))) if isinstance(r, dict) else "")


# --------------------------------------------------------------------------- local bundles

class _HashingReader(io.RawIOBase):
    """Wraps a binary file; hashes exactly the bytes tarfile reads."""

    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, b):
        data = self.f.read(len(b))
        n = len(data)
        b[:n] = data
        self.h.update(data)
        return n


SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini", "__MACOSX", "__pycache__", ".ipynb_checkpoints",
              ".git", ".pytest_cache"}


def _walk_files(root: Path, excludes=()):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_NAMES and d not in excludes)
        for fn in sorted(filenames):
            if fn in SKIP_NAMES or fn.startswith("._") or any(fn.endswith(x) for x in (".pyc", ".pt", ".pth")):
                continue
            yield Path(dirpath) / fn


def write_bundle(entries, out, manifest_name: str, log=sys.stderr) -> dict:
    """Streams (arcname, path) entries into a gzip tar written to `out` (a binary file object) and
    appends a sha256sum-style manifest as the last member. Nothing is staged on disk."""
    import gzip
    lines, total = [], 0
    gz = gzip.GzipFile(fileobj=out, mode="wb", compresslevel=1)      # EEG floats barely compress: favour speed
    with gz, tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for arc, path in entries:
            st = path.stat()
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mtime, ti.mode = st.st_size, int(st.st_mtime), 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = "root"
            with open(path, "rb") as f:
                r = _HashingReader(f)
                tar.addfile(ti, io.BufferedReader(r, 1 << 20))
            lines.append(f"{r.h.hexdigest()}  {arc}")
            total += st.st_size
        blob = ("\n".join(lines) + "\n").encode()
        ti = tarfile.TarInfo(manifest_name)
        ti.size, ti.mtime, ti.mode = len(blob), int(time.time()), 0o644
        tar.addfile(ti, io.BytesIO(blob))
    print(f"bundle: {len(lines)} files, {total / 1e6:.1f} MB uncompressed, manifest {manifest_name}", file=log)
    return dict(files=len(lines), bytes=total, manifest=lines)


def data_entries(bonn: Path, npz: Path, meta: Path, extra=()):
    if not bonn.is_dir():
        raise ApiError(f"Bonn folder not found: {bonn}")
    for f in (npz, meta):
        if not f.is_file():
            raise ApiError(f"file not found: {f}")
    sets = [d.name for d in sorted(bonn.iterdir()) if d.is_dir()]
    if not any(s in sets for s in ("Z", "A", "Set_Z")):
        raise ApiError(f"{bonn} has no Z/ (or A/, Set_Z/) folder: expected bonn/{{Z,O,N,F,S}}/")
    for p in _walk_files(bonn):
        yield "bonn/" + p.relative_to(bonn).as_posix(), p
    yield "chbmit/" + npz.name, npz
    yield "chbmit/" + meta.name, meta
    for p in extra:
        if p.is_file():
            yield "chbmit/" + p.name, p


def _open_out(out: str):
    if out == "-":
        if sys.stdout.isatty():
            raise ApiError("refusing to write a binary archive to the terminal; pipe it to ssh or use --out FILE")
        return sys.stdout.buffer, False
    return open(out, "wb"), True


def cmd_bundle_data(a):
    root = Path(a.data_root).expanduser() if a.data_root else None
    bonn = Path(a.bonn) if a.bonn else (root / "bonn" if root else None)
    npz = Path(a.npz) if a.npz else (root / "chbmit_8ch.npz" if root else None)
    meta = Path(a.meta) if a.meta else (root / "chbmit_8ch_metadata.csv" if root else None)
    if not (bonn and npz and meta):
        raise ApiError("give --data-root (with bonn/, chbmit_8ch.npz, chbmit_8ch_metadata.csv) or "
                       "--bonn, --npz and --meta")
    extra = [npz.with_name("chbmit_8ch_report.json")]
    if not a.out:
        files = list(data_entries(bonn, npz, meta, extra))
        size = sum(p.stat().st_size for _, p in files)
        print(f"{len(files)} files, {size / 1e6:.1f} MB uncompressed:")
        print(f"  bonn/        {sum(1 for x, _ in files if x.startswith('bonn/'))} files from {bonn.as_posix()}")
        for x, p in files:
            if x.startswith("chbmit/"):
                print(f"  {x:40s} {p.stat().st_size / 1e6:8.1f} MB")
        print("\nStream it straight into the pod (Git Bash; no local archive, no extra disk):")
        print(f"  python rp.py bundle-data --data-root \"{root.as_posix() if root else '<dir>'}\" --out - | "
              f"ssh -p <PORT> root@<HOST> \"mkdir -p {POD_WS}/data && tar xzf - -C {POD_WS}/data\"")
        print("Or write one archive (PowerShell-safe; uses local disk) and scp it:")
        print("  python rp.py bundle-data ... --out data_bundle.tgz")
        print(f"  scp -P <PORT> data_bundle.tgz root@<HOST>:{POD_WS}/")
        print(f"  ssh -p <PORT> root@<HOST> \"tar xzf {POD_WS}/data_bundle.tgz -C {POD_WS}/data\"")
        print(f"Then on the pod: bash {POD_RS} check-data   (verifies DATA_MANIFEST.sha256)")
        return
    f, close = _open_out(a.out)
    try:
        write_bundle(data_entries(bonn, npz, meta, extra), f, "DATA_MANIFEST.sha256")
    finally:
        if close:
            f.close()


def code_entries(repo: Path):
    if not (repo / "runpod" / "remote_setup.sh").is_file():
        raise ApiError(f"{repo} is not the study repository (runpod/remote_setup.sh missing)")
    top = repo.name
    for p in _walk_files(repo, excludes={"results", "runs", "data", "pydeps", ".venv", "venv"}):
        yield f"{top}/" + p.relative_to(repo).as_posix(), p


def cmd_bundle_code(a):
    repo = Path(a.repo).resolve()
    f, close = _open_out(a.out)
    try:
        write_bundle(code_entries(repo), f, f"{repo.name}/CODE_MANIFEST.sha256")
    finally:
        if close:
            f.close()


# --------------------------------------------------------------------------- results integrity

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_results(tgz: Path, require_complete=False, out=sys.stdout) -> int:
    """Checks a results archive written by `remote_setup.sh pack`: tarball sha256 against its
    .sha256 sidecar, every member against the MANIFEST.sha256 inside, then summarizes completeness."""
    def say(*x):
        print(*x, file=out)

    problems = []
    side = tgz.with_name(tgz.name + ".sha256")
    digest = _sha256_file(tgz)
    if side.is_file():
        want = side.read_text(encoding="utf-8").split()[0]
        if want != digest:
            problems.append(f"tarball sha256 {digest} != sidecar {want} (transfer corrupted or truncated)")
        else:
            say(f"OK   tarball sha256 matches {side.name}")
    else:
        say(f"WARN no sidecar {side.name}; checking members only")
    manifest, seen, contents = {}, {}, {}
    try:
        with tarfile.open(tgz, "r:gz") as tar:
            members = tar.getmembers()
            for m in members:
                if posixpath.basename(m.name) == "MANIFEST.sha256" and m.isfile():
                    for line in tar.extractfile(m).read().decode().splitlines():
                        if line.strip():
                            h, name = line.split(None, 1)
                            manifest[name.strip().lstrip("*")] = h
            for m in members:
                if posixpath.basename(m.name) == "MANIFEST.sha256" or not (m.isfile() or m.islnk()):
                    continue
                fo = tar.extractfile(m)
                if fo is None:
                    continue
                h = hashlib.sha256()
                for chunk in iter(lambda: fo.read(1 << 20), b""):
                    h.update(chunk)
                seen[m.name] = h.hexdigest()
                if m.name.endswith("jobs.jsonl"):
                    contents[m.name] = tar.extractfile(m).read().decode()
    except (tarfile.TarError, OSError, EOFError) as e:
        problems.append(f"archive unreadable: {e}")
    if not manifest:
        problems.append("MANIFEST.sha256 missing inside the archive")
    else:
        bad = [n for n, h in manifest.items() if seen.get(n) != h]
        missing = [n for n in bad if n not in seen]
        if bad:
            problems.append(f"{len(bad)} member(s) differ from the manifest ({len(missing)} missing), e.g. {bad[:3]}")
        else:
            say(f"OK   {len(manifest)} files match MANIFEST.sha256")
    # completeness summary
    queues = sorted({n.split("/")[1] for n in seen if n.startswith("jobs/") and n.count("/") >= 2})
    for q in queues:
        jobs = [json.loads(x) for x in (contents.get(f"jobs/{q}/jobs.jsonl") or "").splitlines() if x.strip()]
        done = {posixpath.basename(n)[:-5] for n in seen if n.startswith(f"jobs/{q}/done/") and n.endswith(".json")}
        failed = {posixpath.basename(n)[:-5] for n in seen if n.startswith(f"jobs/{q}/failed/") and
                  n.endswith(".json") and n.count("/") == 3} - done
        expect = sum(int(j.get("expect_complete") or 0) for j in jobs)
        merged = sum(1 for n in seen if n.startswith(f"results/{q}/") and n.endswith("/complete.json"))
        runs = sum(1 for n in seen if n.startswith(f"runs/{q}/") and n.endswith("/complete.json"))
        reports = [n for n in seen if n.startswith(f"results/{q}/") and n.endswith(".docx")]
        fin = f"jobs/{q}/finalize.json" in seen
        say(f"queue {q}: jobs {len(done)}/{len(jobs)} done, {len(failed)} failed; complete.json in runs/ {runs}/"
            f"{expect}, in merged results/ {merged}; finalize record: {'yes' if fin else 'no'}; "
            f"Word report(s): {len(reports)}")
        if require_complete and (len(done) < len(jobs) or runs < expect or not fin):
            problems.append(f"queue {q} is incomplete")
    for p in problems:
        say("FAIL", p)
    say("RESULT:", "PASSED" if not problems else "FAILED")
    return 0 if not problems else 1


def cmd_verify_results(a):
    code = 0
    for t in a.tgz:
        print(f"== {t}")
        code |= verify_results(Path(t), a.require_complete)
    sys.exit(code)


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    repo_default = Path(__file__).resolve().parent.parent.as_posix()

    s = sub.add_parser("balance", help="balance, spend, pods, SSH key registration (read-only)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_balance)

    s = sub.add_parser("gpus", help="live GPU stock and prices under a vCPU/RAM filter (read-only)")
    s.add_argument("--cloud", choices=["SECURE", "COMMUNITY", "BOTH"], default="BOTH")
    s.add_argument("--all", action="store_true", help="all GPU types, not only the candidate list")
    s.add_argument("--available", action="store_true",
                   help="only rows in stock for the given filter, cheapest first")
    s.add_argument("--min-vcpu", type=int, help="minimum vCPUs of the machine")
    s.add_argument("--min-ram", type=int, help="minimum RAM (GB) of the machine")
    s.add_argument("--json", action="store_true")
    s.add_argument("--raw", action="store_true", help="with --json: dump the merged raw inventory")
    s.set_defaults(fn=cmd_gpus)

    s = sub.add_parser("create", help="create a GPU pod (DRY RUN unless --yes)")
    s.add_argument("--gpu", required=True, help='exact GPU id, e.g. "NVIDIA RTX A4000" (see gpus --all)')
    s.add_argument("--cloud", choices=["COMMUNITY", "SECURE"], default="SECURE")
    s.add_argument("--image", default=DEFAULT_IMAGE)
    s.add_argument("--container-disk", type=int, default=30, help="GB, wiped on stop")
    s.add_argument("--volume-disk", type=int, default=20, help="GB at /workspace; survives stop, deleted on terminate")
    s.add_argument("--network-volume", help="existing network volume id mounted at /workspace (SECURE only)")
    s.add_argument("--name", default="atcnet-train")
    s.add_argument("--gpu-count", type=int, default=1)
    s.add_argument("--min-vcpu", type=int, default=8, help="minimum vCPUs per GPU (placement filter)")
    s.add_argument("--min-ram", type=int, default=32, help="minimum RAM GB per GPU (placement filter)")
    s.add_argument("--datacenter", action="append", help="preferred data center id (repeatable)")
    s.add_argument("--api", choices=["auto", "v1", "v2"], default="auto",
                   help="auto: v2 for SECURE, v1 for COMMUNITY (public IP) or --interruptible")
    s.add_argument("--interruptible", action="store_true", help="spot pod via REST v1 (no discount observed)")
    s.add_argument("--jupyter", action="store_true", help="also expose 8888/http and start JupyterLab")
    s.add_argument("--pubkey", default=str(DEFAULT_PUBKEY), help="public key injected as PUBLIC_KEY")
    s.add_argument("--allow-duplicate", action="store_true", help="create even if a pod with this name exists")
    s.add_argument("--yes", action="store_true", help="actually send the request (costs money)")
    s.set_defaults(fn=cmd_create)

    s = sub.add_parser("status", help="list pods, or show one pod (read-only)")
    s.add_argument("pod_id", nargs="?")
    s.add_argument("--wait", type=int, default=0, help="poll up to N seconds until RUNNING with direct SSH")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    for name, fn, helptext in [("ssh-cmd", cmd_ssh, "print ssh/scp/tar commands for a pod (read-only)"),
                               ("pull", cmd_pull, "print the commands to pack, download and verify results")]:
        s = sub.add_parser(name, help=helptext)
        s.add_argument("pod_id")
        s.add_argument("--key", default=DEFAULT_PRIVKEY, help="private key file for ssh -i")
        s.add_argument("--queue", default="bonn_atcnet")
        s.add_argument("--repo", default=repo_default, help="local study repository folder")
        s.add_argument("--local-data", default=os.environ.get("EEGDATA", "<data dir>"),
                       help="local folder with bonn/, chbmit_8ch.npz, chbmit_8ch_metadata.csv ($EEGDATA)")
        s.add_argument("--local-results", default="./pod_results")
        s.add_argument("--from-json", help=argparse.SUPPRESS)
        s.set_defaults(fn=fn)

    for name, helptext in [("start", "start a stopped pod"), ("stop", "stop a pod (keeps /workspace)"),
                           ("terminate", "delete a pod and its volume disk")]:
        s = sub.add_parser(name, help=helptext + " (DRY RUN unless --yes)")
        s.add_argument("pod_id")
        s.add_argument("--yes", action="store_true")
        s.set_defaults(fn=(lambda a, _n=name: _lifecycle(a, _n)))

    s = sub.add_parser("bundle-data", help="stream Bonn + CHB-MIT files into one tar.gz with a sha256 manifest")
    s.add_argument("--data-root", default=os.environ.get("EEGDATA"),
                   help="folder with bonn/, chbmit_8ch.npz, chbmit_8ch_metadata.csv (default $EEGDATA)")
    s.add_argument("--bonn")
    s.add_argument("--npz")
    s.add_argument("--meta")
    s.add_argument("--out", help="'-' streams to stdout (pipe to ssh); a path writes a file; omitted: plan only")
    s.set_defaults(fn=cmd_bundle_data)

    s = sub.add_parser("bundle-code", help="stream the study repository (no .git/caches/checkpoints) as tar.gz")
    s.add_argument("--repo", default=repo_default)
    s.add_argument("--out", required=True, help="'-' for stdout or a file path")
    s.set_defaults(fn=cmd_bundle_code)

    s = sub.add_parser("verify-results", help="check a downloaded results archive against its sha256 manifest")
    s.add_argument("tgz", nargs="+")
    s.add_argument("--require-complete", action="store_true", help="also fail if any queue is incomplete")
    s.set_defaults(fn=cmd_verify_results)
    return ap


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass
    a = build_parser().parse_args(argv)
    try:
        a.fn(a)
    except ApiError as e:
        print(f"ERROR: {_scrub(str(e))}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
