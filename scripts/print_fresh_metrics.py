import json
import numpy as np
import pandas as pd

print('=== 1. ATTACK SUMMARY ===')
for atk in ['naive', 'adaptive', 'sinusoidal']:
    data = json.load(open(f'artifacts_v2/results_{atk}.json'))['aggregate']
    rec = data['detect_recall']
    auc = data['clf_auroc']
    dmg = data['damage_pct']
    rf1 = data['rec_classifier']
    rcons = data['rec_conservative']
    roracle = data['rec_oracle']
    print(f"{atk.capitalize():12s}: Recall={rec['mean']:.3f}+/-{rec['std']:.3f}, AUROC={auc['mean']:.3f}+/-{auc['std']:.3f}, Damage={dmg['mean']:.2f}+/-{dmg['std']:.2f}%, Recovery: F1={rf1['mean']:.1f}%, Cons={rcons['mean']:.1f}%, Oracle={roracle['mean']:.1f}%")

print('\n=== 2. BREAK-EVEN SUMMARY ===')
be = json.load(open('artifacts_v2/breakeven.json'))
cl_saved = np.mean([x['saved_per_tp'] for x in be['classifier'] if x.get('saved_per_tp', 0) > 0])
cl_lost = np.mean([x['lost_per_fp'] for x in be['classifier'] if x.get('lost_per_fp', 0) < 0])
cl_prec = np.mean([x['precision'] for x in be['classifier']])
cl_pstar = np.mean([x['breakeven_precision'] for x in be['classifier'] if x.get('breakeven_precision', 0) > 0])

rz_saved = np.mean([x['saved_per_tp'] for x in be['robust_z_rule']])
rz_lost = np.mean([x['lost_per_fp'] for x in be['robust_z_rule']])
rz_prec = np.mean([x['precision'] for x in be['robust_z_rule']])
rz_pstar = np.mean([x['breakeven_precision'] for x in be['robust_z_rule'] if x.get('breakeven_precision', 0) > 0])

print(f"Learned Classifier : Prec={cl_prec:.3f}, $/TP Saved=${cl_saved:.2f}, $/FP Lost=${cl_lost:.2f}, p* Break-Even={cl_pstar:.3f}")
print(f"Robust-z Rule      : Prec={rz_prec:.3f}, $/TP Saved=${rz_saved:.2f}, $/FP Lost=${rz_lost:.2f}, p* Break-Even={rz_pstar:.3f}")

print('\n=== 3. TRANSFER MATRIX (12 rows) ===')
tf = json.load(open('artifacts_v2/results_transfer.json'))
df_tf = pd.DataFrame(tf)
order = [
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
    ('naive+adaptive', 'sinusoidal'),
]

for src, tgt in order:
    sub = df_tf[(df_tf['sources'] == src) & (df_tf['target'] == tgt)]
    print(f"{src:>22s} -> {tgt:<11s} | AUROC={sub['auroc'].mean():.3f} | Prec={sub['precision'].mean():.3f} | Rec={sub['recall'].mean():.3f} | Recovery={sub['rec_conservative'].mean():+6.1f}%")
