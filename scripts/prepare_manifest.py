"""Create a raw-evaluation manifest for official MVTec AD or VisA test splits."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from facetad.paths import DATA
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('dataset',choices=('mvtec','visa'))
p.add_argument('--root',type=Path,required=True,help='Extracted dataset root')
p.add_argument('--output',type=Path)
a=p.parse_args();root=a.root.expanduser().resolve();output=a.output or DATA/'manifests'/f'{a.dataset}.jsonl'
records=[]
def record(image,category,label,masks):
    paths=[image,*masks]
    if any(not x.is_file() for x in paths): raise FileNotFoundError(paths)
    # Absolute paths also work when the user changes DATA_ROOT later.
    return dict(image=str(image.resolve()),category=category,label=label,masks=[str(m.resolve()) for m in masks])
if a.dataset=='mvtec':
    for category in sorted(root.iterdir()):
        if not category.is_dir() or not (category/'test').is_dir(): continue
        for kind in sorted((category/'test').iterdir()):
            if not kind.is_dir(): continue
            for image in sorted(kind.glob('*.png')):
                masks=[] if kind.name=='good' else [category/'ground_truth'/kind.name/(image.stem+'_mask.png')]
                records.append(record(image,category.name,int(kind.name!='good'),masks))
else:
    with (root/'split_csv/1cls.csv').open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        fields=set(reader.fieldnames or [])
        split='split' if 'split' in fields else 'set'
        image_col='image' if 'image' in fields else 'image_path'
        mask_col='mask' if 'mask' in fields else 'mask_path'
        if not {split,image_col,mask_col,'object','label'}.issubset(fields):
            raise ValueError('Unrecognized VisA split CSV columns')
        for row in reader:
            if row[split]!='test': continue
            if row['label'] not in ('normal','anomaly'): raise ValueError(row['label'])
            masks=[root/row[mask_col]] if row[mask_col].strip() else []
            records.append(record(root/row[image_col],row['object'],int(row['label']=='anomaly'),masks))
expected=(1725,15) if a.dataset=='mvtec' else (2162,12)
actual=(len(records),len({r['category'] for r in records}))
if actual!=expected or len({r['image'] for r in records})!=len(records):
    sys.exit(f'Unexpected split: got {actual}, expected {expected} images/classes, with unique IDs. Use the official test split; no manifest written.')
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(''.join(json.dumps(r)+'\n' for r in records))
print(f'Wrote {len(records)} images to {output}. Raw evaluation manifest; not a historical replay-order export.')
