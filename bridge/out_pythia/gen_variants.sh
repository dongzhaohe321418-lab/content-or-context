#!/bin/zsh
# Generator / register variants of the LLM-written probe text, same 24 topic areas (those of probe files w26-w49).
# Topics are read into an array first: codex exec reads stdin, so it must not share the loop's input.
W="$1"; E="$2"; START="${3:-0}"; cd "$E"
topics=("${(@f)$(cat "$W/bridge/out_pythia/probes_gpt/topics.txt")}")
for (( i=START; i<${#topics[@]}; i++ )); do
  t="${topics[$((i+1))]}"
  wiki="Write about 2,500 words of original encyclopedia articles in the plain, factual style of Wikipedia, with section headings written like ' = = History = = '. Topic area: $t (invent plausible specifics freely; do not copy any real article). Output only the article text."
  fic="Write about 2,500 words of original prose fiction: a short story with narration and dialogue, loosely connected to this topic area: $t. Do not write it as an encyclopedia or reference article. Output only the story text."
  ( codex exec --sandbox read-only --skip-git-repo-check --ephemeral -C "$E" -o "$W/bridge/out_pythia/probes_gpt/g$i.txt" "$wiki" < /dev/null > /dev/null 2>&1 ) &
  ( echo "$fic" | claude -p --model claude-opus-5-5 --effort low --tools "" --system-prompt "Answer the user's request." --setting-sources "" --strict-mcp-config --disable-slash-commands --no-session-persistence --output-format json | /usr/bin/python3 -I -c "import json,sys; print(json.load(sys.stdin)['result'])" > "$W/bridge/out_pythia/probes_fiction/f$i.txt" ) &
  if (( (i - START + 1) % 6 == 0 )); then wait; fi
done
wait
