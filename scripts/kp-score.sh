#!/usr/bin/env bash
# Submit one scoring command to a contest compute node.
# Campus queues require both -p and -q. This script sets them together.
#
#   <=8 cores -> kp_interact (max 8h)
#   >8  cores -> kp_run      (max 2h, max 128 cores, 1 node)
#
# Usage:
#   scripts/kp-score.sh -d <problem-dir> -c <cpus-per-task> -t HH:MM:SS -J <name> -- <command...>
#   scripts/kp-score.sh -d 08-blackhole -n 32 -c 1 -t 00:30:00 -J bh -- <mpi-command...>
#
# Logs: logs/score/<name>-<timestamp>.out
# --wait blocks until the job finishes. --dry-run only asks Slurm to validate.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
workdir=""
cpus=""
ntasks=1
timelimit=""
jobname="score"
wait_job=0
dry_run=0

usage() {
  echo "usage: scripts/kp-score.sh -d DIR -c CPUS -t HH:MM:SS [-n TASKS] [-J NAME] [--wait] [--dry-run] -- COMMAND..." >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d) workdir="${2:-}"; shift 2 ;;
    -c) cpus="${2:-}"; shift 2 ;;
    -n) ntasks="${2:-}"; shift 2 ;;
    -t) timelimit="${2:-}"; shift 2 ;;
    -J) jobname="${2:-}"; shift 2 ;;
    --wait) wait_job=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --) shift; break ;;
    *) usage ;;
  esac
done

[[ $# -ge 1 && -n "$workdir" && -n "$cpus" && -n "$timelimit" ]] || usage
[[ "$cpus" =~ ^[0-9]+$ && "$ntasks" =~ ^[0-9]+$ ]] || usage
[[ "$timelimit" =~ ^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] || usage
[[ "$jobname" =~ ^[A-Za-z0-9_.-]+$ ]] || usage

if [[ "$workdir" != /* ]]; then
  workdir="$repo/$workdir"
fi
[[ -d "$workdir" ]] || { echo "missing directory: $workdir" >&2; exit 2; }

total=$((ntasks * cpus))
(( total >= 1 && total <= 128 )) || { echo "cores must be 1..128, got $total" >&2; exit 2; }

if (( total <= 8 )); then
  queue="kp_interact"
  max_seconds=$((8 * 3600))
else
  queue="kp_run"
  max_seconds=$((2 * 3600))
fi

IFS=: read -r hh mm ss <<< "$timelimit"
seconds=$((10#$hh * 3600 + 10#$mm * 60 + 10#$ss))
(( seconds >= 60 && seconds <= max_seconds )) || {
  echo "time $timelimit is outside 00:01:00..$(printf '%02d:00:00' $((max_seconds / 3600))) for $queue" >&2
  exit 2
}

logdir="$repo/logs/score"
mkdir -p "$logdir"
stamp="$(date +%Y%m%d-%H%M%S)"
logfile="$logdir/${jobname}-${stamp}.out"
jobfile="$logdir/${jobname}-${stamp}.sbatch"

{
  printf '%s\n' '#!/bin/bash'
  printf '#SBATCH -p %s\n' "$queue"
  printf '#SBATCH -q %s\n' "$queue"
  printf '#SBATCH -N 1\n'
  printf '#SBATCH -n %s\n' "$ntasks"
  printf '#SBATCH -c %s\n' "$cpus"
  printf '#SBATCH -t %s\n' "$timelimit"
  printf '#SBATCH -J %s\n' "$jobname"
  printf '#SBATCH -o %s\n' "$logfile"
  printf '#SBATCH -e %s\n' "$logfile"
  printf '#SBATCH --chdir=%s\n' "$workdir"
  printf '%s\n' 'set -euo pipefail'
  printf 'export PATH="/vault/public/xflops/bin:${PATH}"\n'
  printf 'cd %q\n' "$workdir"
  printf '%q ' "$@"
  printf '\n'
} > "$jobfile"
chmod 700 "$jobfile"

sbatch_args=()
(( wait_job == 1 )) && sbatch_args+=(--wait)
(( dry_run == 1 )) && sbatch_args+=(--test-only)

submit_out="$(sbatch "${sbatch_args[@]}" "$jobfile")"
echo "$submit_out"
echo "queue=$queue cores=$total log=$logfile"
