#!/bin/zsh
cd "${0:A:h}/../.."
while kill -0 48215 2>/dev/null; do sleep 20; done
.venv/bin/python -I bridge/pb_run.py sweep70free > bridge/out_pb/sweep70free.stdout 2>&1
echo "sweep70free exit $?" >> bridge/out_pb/queue.status
