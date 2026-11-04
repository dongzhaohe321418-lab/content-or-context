#!/bin/zsh
cd "${0:A:h}/../.."
while kill -0 72332 2>/dev/null; do sleep 30; done
.venv/bin/python -I bridge/pb_run.py capacity > bridge/out_pb/capacity.stdout 2>&1
echo "capacity exit $?" >> bridge/out_pb/queue.status
.venv/bin/python -I bridge/pb_mismatch.py capacity pythia-70m pythia-160m 2 1 5 > bridge/out_pb/capacity_swap.stdout 2>&1
echo "capacity swap v3 exit $?" >> bridge/out_pb/queue.status
