#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Multi-process integration test:
#   1. discoveryd + hello_world server + client across processes
#   2. SIGKILL the server; assert its instance disappears via lease TTL
#
# Usage: multi_process.sh <discoveryd> <server> <client> <instance_ls>

set -u

DISCOVERYD=$1
SERVER=$2
CLIENT=$3
INSTANCE_LS=$4

PORT=7797
export DYN_DISCOVERY=127.0.0.1:$PORT
export DYN_LOG=warn

WORKDIR=$(mktemp -d)
DD_PID=""
SRV_PID=""

cleanup() {
  [ -n "$SRV_PID" ] && kill -9 "$SRV_PID" 2>/dev/null
  [ -n "$DD_PID" ] && kill -TERM "$DD_PID" 2>/dev/null
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

"$DISCOVERYD" $PORT >"$WORKDIR/dd.log" 2>&1 &
DD_PID=$!
sleep 1

"$SERVER" >"$WORKDIR/srv.log" 2>&1 &
SRV_PID=$!

# Wait for the instance to register.
i=0
while [ "$i" -lt 50 ]; do
  COUNT=$("$INSTANCE_LS" dynamo backend generate | tail -1)
  [ "$COUNT" = "count=1" ] && break
  i=$((i + 1))
  sleep 0.2
done
[ "$COUNT" = "count=1" ] || fail "server instance never registered (got: $COUNT)"

# End-to-end request.
OUTPUT=$("$CLIENT" 2>/dev/null | grep -v '^\[')
[ "$OUTPUT" = "hello world" ] || fail "unexpected client output: '$OUTPUT'"

# Hard-kill the server: no revoke happens; the lease must lapse via TTL (10s).
kill -9 "$SRV_PID"
SRV_PID=""

i=0
while [ "$i" -lt 90 ]; do
  COUNT=$("$INSTANCE_LS" dynamo backend generate | tail -1)
  [ "$COUNT" = "count=0" ] && break
  i=$((i + 1))
  sleep 0.5
done
[ "$COUNT" = "count=0" ] || fail "instance did not disappear after SIGKILL + TTL"

echo "PASS"
exit 0
