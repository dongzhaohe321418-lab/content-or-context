#!/bin/zsh
cd "${0:A:h}/../.."
while kill -0 61242 2>/dev/null; do sleep 20; done
.venv/bin/python -I bridge/pb_extra.py comm70 pythia-70m pythia-160m 2 1 > bridge/out_pb/comm70_extra.stdout 2>&1
echo "extra(v3 donors) exit $?" >> bridge/out_pb/queue.status
