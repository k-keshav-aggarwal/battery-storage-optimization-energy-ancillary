import json
import pandas as pd

t = json.load(open('artifacts_v2/results_transfer.json'))
df = pd.DataFrame(t)
grouped = df.groupby(['sources', 'target'])[['auroc', 'precision', 'recall', 'rec_conservative']].mean().reset_index()

pairs = [
    ('naive', 'naive'),
    ('naive', 'adaptive'),
    ('naive', 'sinusoidal'),
    ('adaptive', 'naive'),
    ('adaptive', 'adaptive'),
    ('adaptive', 'sinusoidal'),
    ('sinusoidal', 'naive'),
    ('sinusoidal', 'adaptive'),
    ('sinusoidal', 'sinusoidal'),
    ('adaptive+sinusoidal', 'naive'),
    ('naive+sinusoidal', 'adaptive'),
    ('naive+adaptive', 'sinusoidal')
]

for src, tgt in pairs:
    sub = grouped[(grouped['sources'] == src) & (grouped['target'] == tgt)]
    if len(sub) > 0:
        row = sub.iloc[0]
        s_display = src.replace('+', ' + ').title()
        t_display = tgt.title()
        rec_str = f"{row['rec_conservative']:.1f}"
        if row['rec_conservative'] < 0:
            rec_str = f"$-{abs(row['rec_conservative']):.1f}$"
        print(f"    {s_display} & {t_display} & {row['auroc']:.3f} & {row['precision']:.3f} & {row['recall']:.3f} & {rec_str} \\\\")
