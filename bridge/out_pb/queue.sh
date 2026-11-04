#!/bin/zsh
cd "${0:A:h}/../.."
while pgrep -f "pb_run.py sweep70" >/dev/null; do sleep 10; done
for g in comm70 diag textscale; do
  .venv/bin/python -I bridge/pb_run.py $g > bridge/out_pb/$g.stdout 2>&1
  echo "$g exit $?" >> bridge/out_pb/queue.status
done
for g in sweep410_160 sweep160_410; do
  .venv/bin/python -I bridge/pb_run.py $g > bridge/out_pb/$g.stdout 2>&1
  echo "$g exit $?" >> bridge/out_pb/queue.status
done
.venv/bin/python -I bridge/pb_ceiling.py > bridge/out_pb/ceiling.txt 2>&1; echo "ceiling exit $?" >> bridge/out_pb/queue.status
