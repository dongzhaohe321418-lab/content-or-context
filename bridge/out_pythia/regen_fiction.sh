#!/bin/zsh
# Regenerate fiction probe files whose generation was interrupted (the machine slept mid-call).
# usage: regen_fiction.sh <workspace> <empty dir> <index> [<index> ...]
W="$1"; E="$2"; shift 2
cd "$E"
topics=("${(@f)$(cat "$W/bridge/out_pythia/probes_gpt/topics.txt")}")
for i in "$@"; do
  t="${topics[$((i+1))]}"
  fic="Write about 2,500 words of original prose fiction: a short story with narration and dialogue, loosely connected to this topic area: $t. Do not write it as an encyclopedia or reference article. Output only the story text."
  ( echo "$fic" | claude -p --model claude-opus-5-5 --effort low --tools "" --system-prompt "Answer the user's request." \
      --setting-sources "" --strict-mcp-config --disable-slash-commands --no-session-persistence --output-format json \
      | /usr/bin/python3 -I -c "import json,sys; print(json.load(sys.stdin)['result'])" \
      > "$W/bridge/out_pythia/probes_fiction/f$i.txt" ) &
done
wait
