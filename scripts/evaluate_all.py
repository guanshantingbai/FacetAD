"""Evaluate each unique configured target, then summarize eligible domains separately."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from facetad.paths import PROJECT, DATA, OUTPUT
argparse.ArgumentParser(description=__doc__).parse_args()
cfg=json.loads((PROJECT/'configs/config.json').read_text())
extensions={'rsdd','liver_ct','retina_oct','colon_db','cvc300'}
if not extensions.issubset(cfg['datasets']): raise ValueError('Extension names do not match fixed config')
missing=[str(DATA/'manifests'/f'{n}.jsonl') for n in cfg['datasets'] if not (DATA/'manifests'/f'{n}.jsonl').is_file()]
if missing: sys.exit('Prepare all manifests first:\n'+'\n'.join(missing))
OUTPUT.mkdir(parents=True,exist_ok=True)
results=[]
for name in cfg['datasets']:
    dst=OUTPUT/f'{name}.json'
    subprocess.run([sys.executable,str(PROJECT/'scripts/evaluate.py'),name,'--manifest',str(DATA/'manifests'/f'{name}.jsonl'),'--output',str(dst)],check=True)
    results.append(json.loads(dst.read_text()))
summary={}
for group in ('original','extension'):
    chosen=[r for r in results if (r['dataset'] in extensions)==(group=='extension')]
    values={}
    for metric in ('image_auc','image_ap','tpr','pixel_auc','pixel_ap','aupro'):
        valid=[r['macro_percent'][metric] for r in chosen if r['macro_percent'][metric] is not None]
        values[metric]={'mean_percent':sum(valid)/len(valid) if valid else None,'eligible_datasets':len(valid)}
    summary[group]=values
(OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
