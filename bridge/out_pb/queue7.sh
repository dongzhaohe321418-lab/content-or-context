#!/bin/zsh
# v3 donor controls for the groups fitted by queue2/queue3 (their saved bridges), once those queues finish.
cd "${0:A:h}/../.."
S=bridge/out_pb/queue.status
while pgrep -f "out_pb/queue[23].sh" >/dev/null; do sleep 30; done
for g in comm410_160:410m:160m comm160_410:160m:410m; do
  name=${g%%:*}; rest=${g#*:}; a=${rest%%:*}; b=${rest#*:}
  line=$(grep -m1 "chose LA=" bridge/out_pb/$name.stdout)
  la=$(echo $line | sed -E 's/.*LA=([0-9]+), LB=([0-9]+).*/\1/'); lb=$(echo $line | sed -E 's/.*LA=([0-9]+), LB=([0-9]+).*/\2/')
  .venv/bin/python -I bridge/pb_mismatch.py $name pythia-$a pythia-$b $la $lb 5 > bridge/out_pb/${name}_swap.stdout 2>&1
  echo "$name swap v3 (LA=$la LB=$lb) exit $?" >> $S
done
.venv/bin/python -I bridge/pb_mismatch.py textscale pythia-70m pythia-160m 2 1 5 > bridge/out_pb/textscale_swap.stdout 2>&1
echo "textscale swap v3 exit $?" >> $S
