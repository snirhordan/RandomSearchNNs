"""Static check: every b.until()/b.at() cue string must occur in its beat's text."""
import ast
import sys
from pathlib import Path

from narration import SCENES

bad = 0
for path in sys.argv[1:]:
    tree = ast.parse(Path(path).read_text())
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in SCENES]:
        beats = dict(SCENES[cls.name])
        for node in ast.walk(cls):
            if not isinstance(node, ast.With):
                continue
            call = node.items[0].context_expr
            if not (isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "beat"):
                continue
            key = call.args[0].value
            if key not in beats:
                print(f"{cls.name}: unknown beat {key!r}"); bad += 1; continue
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and getattr(sub.func, "attr", "") in ("until", "at", "span")):
                    for a in sub.args:
                        if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value not in beats[key]:
                            print(f"{cls.name}.{key}: cue {a.value!r} not found"); bad += 1
print("cue check:", "OK" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)
