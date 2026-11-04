#!/bin/zsh
# Parallel schedule: text scaling and cross-family pairs start now, alongside comm160_410 (queue2).
cd "${0:A:h}/../.."
S=bridge/out_pb/queue.status
( .venv/bin/python -I bridge/pb_run.py textscale > bridge/out_pb/textscale.stdout 2>&1
  echo "textscale(rerun) exit $?" >> $S
  .venv/bin/python -I bridge/pb_mismatch.py textscale pythia-70m pythia-160m 2 1 5 > bridge/out_pb/textscale_swap.stdout 2>&1
  echo "textscale swap v3 exit $?" >> $S ) &
( for spec in "gpt2_pythia160:openai-community/gpt2:EleutherAI/pythia-160m" \
              "gpt2_qwen05:openai-community/gpt2:Qwen/Qwen2.5-0.5B" \
              "pythia160_smol135:EleutherAI/pythia-160m:HuggingFaceTB/SmolLM2-135M" \
              "opt125_gpt2:facebook/opt-125m:openai-community/gpt2" \
              "pythia160_gpt2:EleutherAI/pythia-160m:openai-community/gpt2"; do
    n=${spec%%:*}; rest=${spec#*:}; a=${rest%%:*}; b=${rest#*:}
    .venv/bin/python -I bridge/pbx.py $n $a $b 0,1 > bridge/out_pb/x_$n.stdout 2>&1
    echo "x_$n exit $?" >> $S
  done ) &
( while pgrep -f "out_pb/queue2.sh" >/dev/null; do sleep 30; done
  line=$(grep -m1 "chose LA=" bridge/out_pb/comm160_410.stdout)
  la=$(echo $line | sed -E 's/.*LA=([0-9]+), LB=([0-9]+).*/\1/'); lb=$(echo $line | sed -E 's/.*LA=([0-9]+), LB=([0-9]+).*/\2/')
  .venv/bin/python -I bridge/pb_mismatch.py comm160_410 pythia-160m pythia-410m $la $lb 5 > bridge/out_pb/comm160_410_swap.stdout 2>&1
  echo "comm160_410 swap v3 (LA=$la LB=$lb) exit $?" >> $S ) &
wait
