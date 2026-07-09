#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
#
# Asserts the Worker hard-exits with code 911 when the app overruns the
# graceful-shutdown window. Note: Unix truncates exit codes to 8 bits, so the
# observable status is 911 % 256 = 143 (same as Dynamo's Rust worker).
#
# Usage: graceful_timeout.sh <hang_worker>

set -u
BINARY=$1

DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT=1 "$BINARY" &
PID=$!
sleep 1

kill -TERM "$PID"
wait "$PID"
STATUS=$?

if [ "$STATUS" -ne 143 ]; then
  echo "FAIL: expected exit status 143 (911 mod 256), got $STATUS" >&2
  exit 1
fi
echo "PASS"
exit 0
