"""Check installation and checkpoint hashes without running inference."""
import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facetad.paths import PROJECT, WEIGHTS, VISUAL, CLIP_CODE, KERNELS, DATA, OUTPUT

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=('visa','mvtec','all'),default='all')
    a=p.parse_args(); errors=[]
    for module in ('torch','torchvision','numpy','scipy','sklearn','PIL','cv2','ftfy','regex','tqdm','thop'):
        try:
            m=importlib.import_module(module)
            print(module,getattr(m,'__version__','installed'))
        except Exception as e: errors.append(f'{module}: {e}')
    try:
        import torch
        if not torch.cuda.is_available(): errors.append('CUDA GPU unavailable')
        else: print('GPU',torch.cuda.get_device_name())
    except ImportError: pass
    cfg=json.loads((PROJECT/'configs/config.json').read_text())
    files=[(VISUAL,cfg['clip_sha'])]
    for source,info in cfg['sources'].items():
        if a.source in ('all',source): files.append((WEIGHTS/(source+'.pt'),info['weights_sha256']))
    for path,expected in files:
        if not path.is_file(): errors.append(f'Missing weights: {path}')
        elif sha(path)!=expected: errors.append(f'SHA mismatch: {path}')
        else: print('SHA OK',path.name)
    if not (CLIP_CODE/'AnomalyCLIP_lib').is_dir(): errors.append(f'Missing pinned AnomalyCLIP checkout: {CLIP_CODE}')
    for name in ('blur.so','histogram.so'):
        if not (KERNELS/name).is_file(): errors.append(f'Missing {name}; run make kernels')
    print('Data root:',DATA,'\nOutput root:',OUTPUT)
    if errors:
        print('\n'.join(errors),file=sys.stderr); return 1
    print('Setup checks passed. Use infer.py to validate loaded CUDA operators.');return 0
if __name__=='__main__': sys.exit(main())
