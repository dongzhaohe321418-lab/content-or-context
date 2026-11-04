#!/bin/zsh
cd "${0:A:h}/../.."
while kill -0 76759 2>/dev/null; do sleep 30; done
n=$(ls bridge/out_pythia/probes_gpt/g*.txt bridge/out_pythia/probes_fiction/f*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "variants: $n probe files present" >> bridge/out_pb/queue.status
.venv/bin/python -I bridge/pb_run.py variants > bridge/out_pb/variants.stdout 2>&1
echo "variants exit $?" >> bridge/out_pb/queue.status
.venv/bin/python -I bridge/pb_mismatch.py variants pythia-70m pythia-160m 2 1 5 > bridge/out_pb/variants_swap.stdout 2>&1
echo "variants swap v3 exit $?" >> bridge/out_pb/queue.status
