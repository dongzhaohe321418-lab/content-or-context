#!/bin/zsh
# v3 donor controls: comm410_160 now; comm160_410 and textscale once queue2/queue3 finish.
cd "${0:A:h}/../.."
S=bridge/out_pb/queue.status
.venv/bin/python -I bridge/pb_mismatch.py comm410_160 pythia-410m pythia-160m 6 1 5 > bridge/out_pb/comm410_160_swap.stdout 2>&1
echo "comm410_160 swap v3 exit $?" >> $S
while pgrep -f "out_pb/queue[23].sh" >/dev/null; do sleep 30; done
line=$(grep -m1 "chose LA=" bridge/out_pb/comm160_410.stdout)
la=$(echo $line | sed -E 's/.*LA=([0-9]+), LB=([0-9]+).*/\1/'); lb=$(echo $line | sed -E 's/.*LA=([0-9]+), LB=([0-9]+).*/\2/')
.venv/bin/python -I bridge/pb_mismatch.py comm160_410 pythia-160m pythia-410m $la $lb 5 > bridge/out_pb/comm160_410_swap.stdout 2>&1
echo "comm160_410 swap v3 (LA=$la LB=$lb) exit $?" >> $S
.venv/bin/python -I bridge/pb_mismatch.py textscale pythia-70m pythia-160m 2 1 5 > bridge/out_pb/textscale_swap.stdout 2>&1
echo "textscale swap v3 exit $?" >> $S
