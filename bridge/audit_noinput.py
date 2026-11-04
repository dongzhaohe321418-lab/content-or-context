"""Flag code in the no-input arm that builds inputs or runs a network forward."""
import json, re, sys
pat = re.compile(r"random\.(rand|normal|uniform|random|integers)|np\.random|default_rng|probe|synthetic|images?\b|draw|stroke|x\s*@\s*W0|@\s*A\[.W0.\]")
for f in sys.argv[1:]:
    for t in map(json.loads, open(f)):
        hits = sorted(set(m.group(0) for m in pat.finditer(t.get("code", ""))))
        if hits:
            print(f, "turn", t["turn"], hits)
