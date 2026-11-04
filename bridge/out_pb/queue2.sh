#!/bin/zsh
# Runs after queue.sh. Re-runs any queue.sh step that did not record a status line (in case lines appended to
# queue.sh while it was running were not picked up), then the cross-scale comm runs.
cd "${0:A:h}/../.."
S=bridge/out_pb/queue.status
while pgrep -f "out_pb/queue.sh" >/dev/null; do sleep 20; done
for g in sweep410_160 sweep160_410; do
  grep -q "^$g exit" $S || { .venv/bin/python -I bridge/pb_run.py $g > bridge/out_pb/$g.stdout 2>&1; echo "$g exit $?" >> $S; }
done
grep -q "^ceiling exit" $S || { .venv/bin/python -I bridge/pb_ceiling.py > bridge/out_pb/ceiling.txt 2>&1; echo "ceiling exit $?" >> $S; }
for g in auto:comm410_160:sweep410_160:410m:160m:0,1,2 auto:comm160_410:sweep160_410:160m:410m:0,1; do
  n=$(echo $g | cut -d: -f2)
  .venv/bin/python -I bridge/pb_run.py $g > bridge/out_pb/$n.stdout 2>&1
  echo "$n exit $?" >> $S
done
