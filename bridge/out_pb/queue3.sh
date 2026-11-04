#!/bin/zsh
cd "${0:A:h}/../.."
while pgrep -f "out_pb/queue2.sh" >/dev/null; do sleep 20; done
.venv/bin/python -I bridge/pb_run.py textscale > bridge/out_pb/textscale.stdout 2>&1
echo "textscale(rerun) exit $?" >> bridge/out_pb/queue.status
