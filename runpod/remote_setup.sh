#!/usr/bin/env bash
# remote_setup.sh - pod bootstrap and concurrent job runner for the ATCNet study notebooks.
#
# Runs on the RunPod pod (Linux, /workspace) and, for smoke tests, locally in Git Bash or Linux with
#   WS=<workspace dir>  PY=<python with papermill>  KERNEL=<registered Jupyter kernel>
# The Python side lives next to this script: rp_queue.py, bench_gpu.py, pod_preflight.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${WS:-/workspace}"; export WS
VENV="$WS/venv"
DATA="$WS/data"
LOGS="$WS/logs"
KERNEL="${KERNEL:-atcnet-venv}"; export KERNEL
if [ -z "${PY:-}" ]; then
  if [ -x "$VENV/bin/python" ]; then PY="$VENV/bin/python"; else PY="$(command -v python3 || command -v python || true)"; fi
fi
QPY="$HERE/rp_queue.py"
MPS_PIPE=/tmp/nvidia-mps
MPS_LOGD=/tmp/nvidia-mps-log
MPS_FLAG=/tmp/rp_mps_on            # in /tmp on purpose: a restarted pod has no MPS daemon
NO_PARAMS='{}'

usage() {
  cat <<'EOF'
usage: bash remote_setup.sh <command> [args]

 setup                               venv (reuses the image's CUDA torch), pinned requirements,
                                     Jupyter kernel, driver/GPU check. Idempotent; re-run after
                                     every pod start (the container disk is wiped on stop).
 check-data [DIR]                    verify DATA_MANIFEST.sha256 (rp.py bundle-data) + layout
 preflight [ARGS]                    pod_preflight.py: GPU, packages, kernel, data, filesystem
 fetch-bonn [DIR]                    download the five official Bonn zips (fallback to uploading)
 prep-notebook NB [OUT]              tag the config cell 'parameters' (no source change)
 calibrate [NB] [MAXP] [SEC] [PRICE] [bonn|chbmit] [both|off|on]
                                     sweep 1..MAXP concurrent workers with CUDA MPS off and on;
                                     prints the worker count and mode to launch with
 make-bonn-jobs Q NB [DATA_DIR] [PARAMS_JSON]
                                     one job per PREVIOUS_32 combination, 5->4->3->2 classes
 make-chbmit-jobs Q NB [NPZ] [META] [FOLDS_PER_JOB] [PARAMS_JSON]
                                     one job per chunk of LOPO test units (protocol fold order)
 add-job Q ID NB PARAMS_JSON [EXPECT_COMPLETE]
 launch Q N [tmux|nohup] [--more]    start N workers (refuses while the queue has live
                                     processes, unless --more adds workers to a running queue)
 progress Q                          jobs, fits, rate, ETA, running/failed/interrupted/orphaned
 util [INTERVAL] [COUNT]             GPU/CPU/RAM snapshot(s)
 stop-workers Q                      graceful: workers exit after their current job
 kill-workers Q                      kill this queue's workers, kernels and finalize run only
 requeue Q [--force] [IDS...]        release interrupted/failed jobs whose processes are dead;
                                     refuses while workers run unless --force; never live jobs
 merge Q                             hard-link done jobs' results into results/Q/study_<ID>
 finalize Q [--force] [tmux]         merge + one notebook pass over completed jobs that only
                                     reuses fits (Bonn) or aggregates (CHB-MIT); never trains
 autostop Q [--stop-pod] [--interval S] [--max-hours H] [--dry-run] [--no-finalize]
                                     detached watchdog: when Q is idle, finalize + pack; with
                                     --stop-pod it then stops THIS pod via the API (opt-in)
 pack [Q] [--no-checkpoints] [--keep N]
                                     results archive + sha256 sidecars in $WS/packs (safe mid-run)
 mps-on | mps-off                    CUDA MPS daemon for all later workers (see calibrate)

environment: WS (default /workspace), PY, KERNEL (default atcnet-venv), WORKER_THREADS (1),
             STAGGER (seconds between worker starts, 10), FOLDS_CSV (CHB-MIT fold order)
EOF
}

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_py() { [ -n "$PY" ] || die "no python found; set PY"; }
q_py() { need_py; "$PY" "$QPY" "$@"; }

eff_cpus() {  # vCPUs granted to this container
  if [ -n "${RUNPOD_CPU_COUNT:-}" ]; then echo "$RUNPOD_CPU_COUNT"; return; fi
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    local q p
    read -r q p < /sys/fs/cgroup/cpu.max || true
    if [ "${q:-max}" != "max" ] && [ -n "${p:-}" ]; then echo $(( (q + p - 1) / p )); return; fi
  fi
  nproc 2>/dev/null || echo 1
}

worker_env() {  # environment of every worker and finalize run
  local t="${WORKER_THREADS:-1}"
  WENV=("OMP_NUM_THREADS=$t" "MKL_NUM_THREADS=$t" "OPENBLAS_NUM_THREADS=$t" "NUMEXPR_NUM_THREADS=$t"
        MPLBACKEND=Agg PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
        PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 "WS=$WS" "KERNEL=$KERNEL")
  # PIP_NO_INDEX: a missing package makes the notebook's own pip install fail fast instead of
  # changing versions (and thus STUDY_ID) in the middle of a study.
  if [ -f "$MPS_FLAG" ]; then WENV+=("CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE" "CUDA_MPS_LOG_DIRECTORY=$MPS_LOGD"); fi
}

has_tmux() { command -v tmux >/dev/null 2>&1; }

# ----------------------------------------------------------------------------------------- setup
cmd_setup() {
  [ "$(uname -s)" = Linux ] || die "setup prepares the Linux pod; for a local smoke test set PY and KERNEL instead"
  mkdir -p "$WS/code" "$DATA" "$WS/jobs" "$WS/runs" "$WS/results" "$LOGS" "$WS/packs"
  export PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
  if ! has_tmux && command -v apt-get >/dev/null; then
    log "apt: installing tmux (container disk; repeated after each pod start)"
    (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null) \
      || log "WARNING: apt install failed; use 'launch Q N nohup'"
  fi
  if [ ! -x "$VENV/bin/python" ]; then
    log "creating $VENV (system site packages: reuses the image's CUDA torch)"
    if ! python3 -m venv --system-site-packages "$VENV" 2>/dev/null; then
      local pyv
      pyv=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
      (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "python${pyv}-venv" >/dev/null) || true
      python3 -m venv --system-site-packages "$VENV"
    fi
  fi
  PY="$VENV/bin/python"
  local cons=()
  if [ -f "$WS/constraints.txt" ]; then cons=(-c "$WS/constraints.txt"); log "installing with $WS/constraints.txt"; fi
  log "pip install -r requirements-runpod.txt (into $VENV, persistent)"
  "$PY" -m pip install -q ${cons[@]+"${cons[@]}"} -r "$HERE/requirements-runpod.txt"
  if [ ! -f "$WS/constraints.txt" ]; then
    "$PY" -m pip freeze | grep -v -e '^-e ' -e ' @ ' > "$WS/constraints.txt" || true
    log "wrote $WS/constraints.txt: copy it to /workspace of any replacement pod BEFORE its setup"
  fi
  "$PY" -m ipykernel install --prefix "$VENV" --name "$KERNEL" --display-name "ATCNet study (venv)" >/dev/null
  "$PY" -m pip freeze > "$LOGS/pip_freeze.txt"
  if command -v nvidia-smi >/dev/null; then
    local drv
    drv=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
    log "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1); driver CUDA $drv"
    if [ -n "$drv" ] && [ "$(printf '%s\n12.8\n' "$drv" | sort -V | head -1)" != "12.8" ]; then
      log "WARNING: host driver supports CUDA $drv < 12.8; the cu128 image may not run: stop this pod and pick another host"
    fi
  else
    log "WARNING: nvidia-smi not found (no GPU visible)"
  fi
  "$PY" - <<'PYEOF'
import platform, torch
print("python", platform.python_version(), "| torch", torch.__version__, "| CUDA build", torch.version.cuda,
      "| cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(8, 1, 4097, device="cuda"); c = torch.nn.Conv1d(1, 16, 64).cuda(); c(x).sum().backward()
    torch.cuda.synchronize(); print("CUDA conv forward/backward OK on", torch.cuda.get_device_name(0))
PYEOF
  local mem=""
  if command -v free >/dev/null; then mem=$(free -g | awk '/Mem:/{print $2" GB RAM, "$7" GB available"}'); fi
  log "vCPUs granted: $(eff_cpus); $mem"
  df -h "$WS" / | sed 's/^/  /'
  log "setup done. Kernel '$KERNEL' -> $PY. Next: bash $0 check-data && bash $0 preflight"
}

# ------------------------------------------------------------------------------------------ data
cmd_check_data() {
  local d="${1:-$DATA}"
  [ -d "$d" ] || die "no such dir $d"
  if [ -f "$d/DATA_MANIFEST.sha256" ]; then
    if (cd "$d" && sha256sum -c --quiet DATA_MANIFEST.sha256); then
      log "DATA_MANIFEST.sha256: all $(grep -c . "$d/DATA_MANIFEST.sha256") files match"
    else
      die "DATA_MANIFEST.sha256 mismatch: re-upload the data bundle"
    fi
  else
    log "no DATA_MANIFEST.sha256 in $d (not uploaded with rp.py bundle-data): checking layout only"
  fi
  local b="$d/bonn" s a f n
  if [ -d "$b" ]; then
    for s in Z O N F S; do
      case $s in Z) a=A;; O) a=B;; N) a=C;; F) a=D;; S) a=E;; esac
      f=""
      for cand in "$b/$s" "$b/$a" "$b/Set_$s"; do if [ -d "$cand" ]; then f="$cand"; break; fi; done
      if [ -z "$f" ]; then printf '  bonn set %s: MISSING\n' "$s"; continue; fi
      n=$(find "$f" -type f -iname '*.txt' -not -path '*__MACOSX*' | wc -l)
      printf '  bonn set %s: %-44s %4s txt files\n' "$s" "$f" "$n"
    done
  else
    log "no Bonn folder at $b"
  fi
  for f in "$d/chbmit/chbmit_8ch.npz" "$d/chbmit/chbmit_8ch_metadata.csv"; do
    if [ -f "$f" ]; then printf '  %-60s %s\n' "$f" "$(du -h "$f" | cut -f1)"; else printf '  %-60s MISSING\n' "$f"; fi
  done
  echo "  (expected: 100 txt files per Bonn set; run 'preflight' for the full content checks)"
}

cmd_preflight() {
  need_py
  mkdir -p "$LOGS"
  "$PY" "$HERE/pod_preflight.py" --kernel "$KERNEL" --json "$LOGS/preflight_$(date +%Y%m%d_%H%M%S).json" "$@"
}

cmd_fetch_bonn() {
  local dst="${1:-$DATA/bonn}" z="$DATA/bonn_zips"
  mkdir -p "$dst" "$z"
  # official page: https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/
  local spec="z:21874:591469 o:21872:642112 n:21871:587464 f:21870:596691 s:21875:787492" item s id size url got
  : > "$dst/SOURCE_sha256.txt"
  for item in $spec; do
    IFS=: read -r s id size <<<"$item"
    url="https://www.ukbonn.de/site/assets/files/$id/$s.zip"
    [ -s "$z/$s.zip" ] || curl -fsSL --retry 4 --retry-delay 3 -A "Mozilla/5.0" -o "$z/$s.zip" "$url"
    got=$(wc -c < "$z/$s.zip" | tr -d ' ')
    [ "$got" = "$size" ] || log "WARNING: $s.zip is $got bytes, expected $size (changed upstream?)"
    "$PY" -c "import zipfile,sys; zf=zipfile.ZipFile(sys.argv[1]); [zf.extract(m, sys.argv[2]) for m in zf.namelist() if not m.startswith('__MACOSX')]" "$z/$s.zip" "$dst"
    echo "$url $(sha256sum "$z/$s.zip" | cut -d' ' -f1) $got" >> "$dst/SOURCE_sha256.txt"
  done
  cmd_check_data "$(dirname "$dst")"
}

default_nb() {
  local f
  for f in "$WS"/code/seizure_study/notebooks/Bonn_ATCNet_5_to_2_Class_Validation.ipynb \
           "$WS"/code/seizure_study/notebooks/*ATCNet*.ipynb "$WS"/code/pod_*ATCNet*.ipynb; do
    if [ -f "$f" ]; then echo "$f"; return; fi
  done
}

# ----------------------------------------------------------------------------------- calibration
cmd_calibrate() {
  need_py
  local nb="${1:-$(default_nb)}" maxp="${2:-$(eff_cpus)}" sec="${3:-40}" price="${4:-0}" ds="${5:-bonn}" mps="${6:-both}"
  local plist="" p dev=cuda stamp
  [ -f "$nb" ] || die "notebook not found (arg 1): '$nb'"
  for p in 1 2 3 4 6 8 10 12 16 20 24; do
    if [ "$p" -le "$maxp" ]; then plist="$plist,$p"; fi
  done
  if ! command -v nvidia-smi >/dev/null; then dev=cpu; mps=off; log "no GPU: CPU calibration only"; fi
  stamp=$(date +%Y%m%d_%H%M%S)
  mkdir -p "$LOGS"
  log "calibration: $ds, P=${plist#,}, ${sec}s per point, MPS $mps"
  # RP_BENCH_ARGS: extra bench args, e.g. --model-options '{"tcn_depth": 3}' or --plot-every 100
  # shellcheck disable=SC2086
  "$PY" "$HERE/bench_gpu.py" "$nb" --dataset "$ds" --procs "${plist#,}" --seconds "$sec" --price "$price" \
    --mps "$mps" --device "$dev" --out "$LOGS/calibration_${ds}_$stamp.json" ${RP_BENCH_ARGS:-} \
    | tee "$LOGS/calibration_${ds}_$stamp.txt"
}

cmd_mps_on() {
  command -v nvidia-cuda-mps-control >/dev/null || die "nvidia-cuda-mps-control not in this image"
  mkdir -p "$MPS_PIPE" "$MPS_LOGD"
  CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOGD" nvidia-cuda-mps-control -d \
    || die "MPS daemon did not start (not permitted in this container?): run workers without MPS"
  touch "$MPS_FLAG"
  log "MPS on: workers launched from now on share the GPU through MPS (mps-off to stop)"
}

cmd_mps_off() {
  if command -v nvidia-cuda-mps-control >/dev/null; then
    echo quit | CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control 2>/dev/null || true
  fi
  rm -f "$MPS_FLAG"
  log "MPS off"
}

# ----------------------------------------------------------------------------------------- queue
cmd_launch() {
  need_py
  local q="${1:?queue}" n="${2:?number of workers}" mode="${3:-tmux}" more="${4:-}" stagger="${STAGGER:-10}"
  local start i delay cmd
  if [ "$mode" = "--more" ]; then more="--more"; mode=tmux; fi
  [ -f "$WS/jobs/$q/jobs.jsonl" ] || die "no queue $q (make-bonn-jobs / make-chbmit-jobs first)"
  if [ "$more" = "--more" ]; then
    start=$(q_py next-wid "$q")
  else
    if ! q_py live "$q" >/dev/null; then
      q_py live "$q" || true
      die "queue $q still has live processes (above). Add workers with: launch $q N $mode --more ; or kill-workers $q first"
    fi
    start=1
  fi
  rm -f "$WS/jobs/$q/STOP"
  mkdir -p "$LOGS/$q"
  if [ "$n" -gt "$(eff_cpus)" ]; then log "WARNING: $n workers > $(eff_cpus) vCPUs"; fi
  worker_env
  if [ "$mode" = tmux ] && ! has_tmux; then log "tmux not available: using nohup"; mode="nohup"; fi
  for i in $(seq "$start" $((start + n - 1))); do
    delay=$(( (i - start) * stagger ))
    if [ "$mode" = tmux ]; then
      cmd="$(printf '%q ' env "${WENV[@]}" "$PY" "$QPY" work "$q" "$i" "$delay") 2>&1 | tee -a $(printf '%q' "$LOGS/$q/worker_$i.log")"
      if tmux has-session -t "=rp_$q" 2>/dev/null; then
        tmux new-window -t "=rp_$q" -n "w$i" "$cmd"
      else
        tmux new-session -d -s "rp_$q" -n "w$i" "$cmd"
      fi
    elif command -v setsid >/dev/null; then
      nohup setsid env "${WENV[@]}" "$PY" "$QPY" work "$q" "$i" "$delay" >> "$LOGS/$q/worker_$i.log" 2>&1 &
    else
      nohup env "${WENV[@]}" "$PY" "$QPY" work "$q" "$i" "$delay" >> "$LOGS/$q/worker_$i.log" 2>&1 &
    fi
  done
  log "started workers $start..$((start + n - 1)) for $q ($mode, ${stagger}s apart); logs: $LOGS/$q/worker_*.log"
  if [ "$mode" = tmux ]; then log "attach: tmux attach -t rp_$q (detach: Ctrl-b d)"; fi
  log "watch: bash $0 progress $q ; bash $0 util 5 3"
}

cmd_stop_workers() { touch "$WS/jobs/${1:?queue}/STOP"; log "STOP requested for $1: workers exit after their current job"; }

cmd_kill_workers() {
  local q="${1:?queue}"
  q_py kill "$q" || true
  if has_tmux && tmux has-session -t "=rp_$q" 2>/dev/null; then
    tmux kill-session -t "=rp_$q" && log "closed tmux session rp_$q"
  fi
  if q_py live "$q" >/dev/null; then
    log "no live processes left for $q. Next: bash $0 requeue $q"
  else
    q_py live "$q" || true
    die "processes of $q survived; inspect them before requeue"
  fi
}

cmd_finalize() {
  need_py
  local q="${1:?queue}"; shift
  local mode=fg a args=()
  for a in "$@"; do if [ "$a" = tmux ]; then mode=tmux; else args+=("$a"); fi; done
  mkdir -p "$LOGS/$q"
  worker_env
  if [ "$mode" = tmux ] && has_tmux; then
    tmux new-session -d -s "rp_${q}_final" \
      "$(printf '%q ' env "${WENV[@]}" "$PY" "$QPY" finalize "$q" ${args[@]+"${args[@]}"}) 2>&1 | tee -a $(printf '%q' "$LOGS/$q/finalize.out")"
    log "finalize running in tmux session rp_${q}_final (log $LOGS/$q/finalize.out)"
  else
    env "${WENV[@]}" "$PY" "$QPY" finalize "$q" ${args[@]+"${args[@]}"} 2>&1 | tee -a "$LOGS/$q/finalize.out"
  fi
}

cmd_autostop() {
  need_py
  local q="${1:?queue}"; shift
  local a fg=0 args=()
  for a in "$@"; do if [ "$a" = --foreground ]; then fg=1; else args+=("$a"); fi; done
  [ -f "$WS/jobs/$q/meta.json" ] || die "no queue $q"
  mkdir -p "$LOGS/$q"
  worker_env
  if [ "$fg" = 1 ]; then
    env "${WENV[@]}" "$PY" "$QPY" autostop "$q" ${args[@]+"${args[@]}"}
  elif has_tmux; then
    tmux new-session -d -s "rp_${q}_autostop" \
      "$(printf '%q ' env "${WENV[@]}" "$PY" "$QPY" autostop "$q" ${args[@]+"${args[@]}"}) >> $(printf '%q' "$LOGS/$q/autostop.out") 2>&1"
    log "autostop watchdog running in tmux session rp_${q}_autostop; log $LOGS/$q/autostop.log"
  else
    nohup env "${WENV[@]}" "$PY" "$QPY" autostop "$q" ${args[@]+"${args[@]}"} >> "$LOGS/$q/autostop.out" 2>&1 &
    log "autostop watchdog started (nohup); log $LOGS/$q/autostop.log"
  fi
  case " ${args[*]:-} " in
    *" --stop-pod "*) log "it WILL stop this pod when $q is idle (after finalize + pack); kill-workers $q cancels it" ;;
    *) log "it will finalize + pack only; add --stop-pod to also stop the pod" ;;
  esac
}

cmd_util() {
  local interval="${1:-0}" count="${2:-1}" k
  for k in $(seq 1 "$count"); do
    echo "=== $(date '+%F %T')  vCPUs=$(eff_cpus)  load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo '?')"
    if command -v nvidia-smi >/dev/null; then
      nvidia-smi --query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv | sed 's/^/  /'
    fi
    if [ -r /proc/stat ]; then
      "${PY:-python3}" - <<'PYEOF'
import time
def snap():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]
t1, i1 = snap(); time.sleep(1); t2, i2 = snap()
print(f"  CPU busy (all host cores visible): {100 * (1 - (i2 - i1) / max(1, t2 - t1)):.0f}%")
PYEOF
    fi
    if command -v free >/dev/null; then free -g | awk '/Mem:/{printf "  RAM: %s GB used / %s GB total\n", $3, $2}'; fi
    ps -eo pid,pcpu,pmem,etime,args --sort=-pcpu 2>/dev/null | head -8 | cut -c1-150 | sed 's/^/  /' || true
    df -h "$WS" 2>/dev/null | tail -1 | awk '{print "  disk "$6": "$4" free of "$2}' || true
    if [ "$k" -lt "$count" ]; then sleep "$interval"; fi
  done
  echo "  hint: GPU util well below 100% while worker CPUs are busy => CPU/launch bound; more workers help only while vCPUs remain"
}

# ------------------------------------------------------------------------------------------ main
c="${1:-}"; shift || true
case "$c" in
  setup) cmd_setup "$@" ;;
  check-data) cmd_check_data "$@" ;;
  preflight) cmd_preflight "$@" ;;
  fetch-bonn) cmd_fetch_bonn "$@" ;;
  prep-notebook) q_py prep-notebook "$@" ;;
  calibrate) cmd_calibrate "$@" ;;
  make-bonn-jobs) q_py make-bonn "${1:?queue}" "${2:?notebook}" "${3:-$DATA/bonn}" "${4:-$NO_PARAMS}" ;;
  make-chbmit-jobs) q_py make-chbmit "${1:?queue}" "${2:?notebook}" "${3:-$DATA/chbmit/chbmit_8ch.npz}" \
                      "${4:-$DATA/chbmit/chbmit_8ch_metadata.csv}" "${5:-1}" "${6:-$NO_PARAMS}" ;;
  add-job) q_py add "$@" ;;
  launch) cmd_launch "$@" ;;
  progress) q_py progress "${1:?queue}" ;;
  util) cmd_util "$@" ;;
  stop-workers) cmd_stop_workers "$@" ;;
  kill-workers) cmd_kill_workers "$@" ;;
  requeue) q_py requeue "$@" ;;
  merge) q_py merge "${1:?queue}" ;;
  finalize) cmd_finalize "$@" ;;
  autostop) cmd_autostop "$@" ;;
  pack) q_py pack "$@" ;;
  mps-on) cmd_mps_on ;;
  mps-off) cmd_mps_off ;;
  help|-h|--help) usage ;;
  *) usage; exit 1 ;;
esac
