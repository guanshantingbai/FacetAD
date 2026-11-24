"""Compare refactored inference/evaluation with a preserved public-release checkout.

Requires the original benchmark caches and images; not part of the quickstart.
Writes only a numerical receipt to the requested output directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--cache-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--worker',choices=('old','new'))
p.add_argument('--target',choices=('mvtec','visa'))
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
if a.worker:
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn.functional import normalize
    if a.worker=='old':
        sys.path.insert(0,str(a.baseline/'release'))
        import runtime as rt
        import evaluate as ev
    else:
        sys.path.insert(0,str(ROOT))
        from facetad import runtime as rt,evaluate as ev
    cache=a.cache_root/a.target
    rows=json.loads((cache/'rows.json').read_text())
    member=rt.Ensemble(rt.source_for(a.target))
    g,z=[normalize(torch.from_numpy(np.load(cache/(n+'.npy'),mmap_mode='r')[:8].copy()).cuda().float(),dim=-1) for n in ('global','patch')]
    ims=[]
    for r in rows[:8]:
        with Image.open(r['image']) as im: ims.append(rt.TRANS.transforms[0](im.convert('RGB')))
    cp,ch=member.predict_tokens(g,z,ims)
    with Image.open(rows[0]['image']) as im: rp,rh=member.predict([im])
    np.savez_compressed(a.output/f'{a.worker}-{a.target}.npz',cached_probability=cp,cached_heat=ch.cpu().numpy(),raw_probability=rp,raw_heat=rh.cpu().numpy())
    del member
    torch.cuda.empty_cache()
    selected=[];cat=rows[0]['category']
    for y in (0,1): selected.extend([r for r in rows if r['category']==cat and r['y']==y][:2])
    assert len(selected)==4
    manifest=a.output/f'{a.worker}-{a.target}.jsonl'
    def path(s):
        if a.worker=='new' and 'FACETAD_DATA_ROOT' in os.environ:
            return str(Path(s).relative_to(Path(os.environ['FACETAD_DATA_ROOT'])))
        return s
    manifest.write_text(''.join(json.dumps(dict(image=path(r['image']),category=r['category'],label=r['y'],masks=[path(m) for m in r['masks']]))+'\n' for r in selected))
    result=ev.evaluate(a.target,manifest)
    (a.output/f'{a.worker}-{a.target}.json').write_text(json.dumps(result,indent=2))
else:
    import numpy as np
    results={}
    for target in ('mvtec','visa'):
        for worker in ('old','new'):
            subprocess.run([sys.executable,__file__,'--baseline',str(a.baseline),'--cache-root',str(a.cache_root),'--output',str(a.output),'--target',target,'--worker',worker],check=True)
        with np.load(a.output/f'old-{target}.npz') as old,np.load(a.output/f'new-{target}.npz') as new:
            errors={k:float(np.max(np.abs(old[k]-new[k]))) for k in old.files}
        assert all(v==0 for v in errors.values()),errors
        old=json.loads((a.output/f'old-{target}.json').read_text());new=json.loads((a.output/f'new-{target}.json').read_text())
        assert old==new,(old,new)
        results[target]=dict(cached_images=8,raw_images=1,evaluation_subset_images=4,max_abs=errors,six_metrics_equal=True)
    (a.output/'regression.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
