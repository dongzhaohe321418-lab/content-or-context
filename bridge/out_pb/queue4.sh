#!/bin/zsh
# Cross-family pairs (pbx.py), after all earlier queues.
cd "${0:A:h}/../.."
while pgrep -f "out_pb/queue[23].sh" >/dev/null || pgrep -f "sweep70free|pb_mismatch.py" >/dev/null; do sleep 30; done
for spec in "gpt2_pythia160:openai-community/gpt2:EleutherAI/pythia-160m" \
            "gpt2_qwen05:openai-community/gpt2:Qwen/Qwen2.5-0.5B" \
            "pythia160_smol135:EleutherAI/pythia-160m:HuggingFaceTB/SmolLM2-135M" \
            "opt125_gpt2:facebook/opt-125m:openai-community/gpt2" \
            "pythia160_gpt2:EleutherAI/pythia-160m:openai-community/gpt2"; do
  n=${spec%%:*}; rest=${spec#*:}; a=${rest%%:*}; b=${rest#*:}
  .venv/bin/python -I bridge/pbx.py $n $a $b 0,1 > bridge/out_pb/x_$n.stdout 2>&1
  echo "x_$n exit $?" >> bridge/out_pb/queue.status
done
