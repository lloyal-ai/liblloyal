#!/usr/bin/env python3
"""Tabulate two Bench JSONL files item by item and apply the stop rule:
compare.py <baseline.jsonl> <candidate.jsonl>

A line without an "item" key (the baseline header) is ignored. The stop rule for a
refactor is a p50 regression over 10 percent on any decode, prefill or image row;
a gated row missing from the candidate is a stop too. The exit status is the gate:
0 when every gated row holds, 1 when any stops it. The ratio is judged unrounded
and printed to two places; a stopped row is marked STOP."""
import json, sys
GATED = ('decode_', 'prefill', 'image_')
THRESHOLD = 1.10
def load(p):
    rows = {}
    for line in open(p):
        line = line.strip()
        if not line or not line.startswith('{'):
            continue
        d = json.loads(line)
        if 'item' not in d:
            continue
        rows[d['item']] = d
    return rows
a, b = load(sys.argv[1]), load(sys.argv[2])
la = next(iter(a.values()))['label'] if a else 'a'
lb = next(iter(b.values()))['label'] if b else 'b'
gated = lambda k: k.startswith(GATED)
stops = []
print(f"| item | unit | {la} p50 | {lb} p50 | ratio | {la} p90 | {lb} p90 |")
print("| -- | -- | -- | -- | -- | -- | -- |")
for k in a:
    if k not in b:
        print(f"| {k} | {a[k]['unit']} | {a[k]['p50']:.2f} | missing | {'STOP' if gated(k) else ''} | | |")
        if gated(k):
            stops.append(f"{k}: missing from the candidate")
        continue
    ra = a[k]['p50']; rb = b[k]['p50']
    ratio = (rb / ra) if ra else float('nan')
    if gated(k) and ratio > THRESHOLD:
        stops.append(f"{k}: {ratio:.4f}x at p50")
    extra = ''
    for f in ('tok_per_s', 'cells', 'advance', 'grammar_bytes', 'tokens'):
        if f in a[k] or f in b[k]:
            extra += f" {f}={a[k].get(f, '?')}/{b[k].get(f, '?')}"
    mark = ' STOP' if gated(k) and ratio > THRESHOLD else ''
    print(f"| {k}{(' (' + extra.strip() + ')') if extra else ''} | {a[k]['unit']} | {ra:.2f} | {rb:.2f} | {ratio:.2f}x{mark} | {a[k]['p90']:.2f} | {b[k]['p90']:.2f} |")
if stops:
    print()
    print("Stop rule: " + "; ".join(stops))
    sys.exit(1)
