#!/usr/bin/env bash

set -u

if [ "$#" -ne 1 ]; then
	echo "usage: $0 <tasks.txt>" >&2
	exit 2
fi

TASKS=$1
BIN=$(dirname "$0")/md5fastcoll

if [ ! -r "$TASKS" ]; then
	echo "$0: cannot read task file: $TASKS" >&2
	exit 1
fi

if [ ! -x "$BIN" ]; then
	echo "$0: $BIN not found or not executable; run 'make all' first" >&2
	exit 1
fi

if [ -n "${MINICLASH_WORKERS:-}" ]; then
	workers=$MINICLASH_WORKERS
else
	allowed=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)
	workers=0
	IFS=',' read -ra parts <<< "${allowed:-}"
	for part in "${parts[@]}"; do
		[ -n "$part" ] || continue
		case "$part" in
			*-*)
				lo=${part%-*}
				hi=${part#*-}
				workers=$((workers + hi - lo + 1))
				;;
			*)
				workers=$((workers + 1))
				;;
		esac
	done
	if [ "$workers" -lt 1 ]; then
		workers=$(nproc 2>/dev/null || echo 1)
	fi
fi
if [ "$workers" -gt 32 ]; then
	workers=32
fi
if [ "$workers" -lt 1 ]; then
	workers=1
fi

exec "$BIN" --threads "$workers" --tasks "$TASKS"
