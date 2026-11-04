#!/bin/zsh
# usage: run_toolfree.sh <pair> <run> <effort>
cd /private/tmp/claude-501/-Users-ericdong-Library-Application-Support-Claude-scratch-workspaces-74ef214a-bb0b-45f3-a0dd-dab068ffcbe9-c2cff56f-cccf-49f5-bf0f-c2a959dccdde-scratch-2026-11-02-a4ee54/2a9261e4-5e20-43c0-ae1d-9fe6adf27cb8/scratchpad/empty
claude -p --model claude-opus-5-5 --effort $3 --tools "" --system-prompt "Answer the user's request." \
  --setting-sources "" --strict-mcp-config --disable-slash-commands --no-session-persistence --output-format json \
  < "/Users/ericdong/Library/Application Support/Claude/scratch-workspaces/74ef214a-bb0b-45f3-a0dd-dab068ffcbe9/c2cff56f-cccf-49f5-bf0f-c2a959dccdde/scratch-2026-11-02-a4ee54/bridge/out/prompt_$1.txt" > "/Users/ericdong/Library/Application Support/Claude/scratch-workspaces/74ef214a-bb0b-45f3-a0dd-dab068ffcbe9/c2cff56f-cccf-49f5-bf0f-c2a959dccdde/scratch-2026-11-02-a4ee54/bridge/out/llm/$1_toolfree-$3_r$2.json" 2> "/Users/ericdong/Library/Application Support/Claude/scratch-workspaces/74ef214a-bb0b-45f3-a0dd-dab068ffcbe9/c2cff56f-cccf-49f5-bf0f-c2a959dccdde/scratch-2026-11-02-a4ee54/bridge/out/llm/$1_toolfree-$3_r$2.err"
