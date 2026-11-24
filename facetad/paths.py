"""Filesystem configuration only; scientific settings remain in configs/config.json."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
def setting(name, default, legacy=None):
    return Path(os.environ.get(name, os.environ.get(legacy, str(default)) if legacy else str(default))).expanduser().resolve()

DATA = setting('FACETAD_DATA_ROOT', PROJECT / 'data')
OUTPUT = setting('FACETAD_OUTPUT_ROOT', PROJECT / 'outputs')
WEIGHTS = setting('FACETAD_WEIGHTS_ROOT', PROJECT / 'weights', 'R1_WEIGHTS')
KERNELS = setting('FACETAD_KERNELS', PROJECT / 'build', 'R1_KERNELS')
CLIP_CODE = setting('FACETAD_CLIP_CODE', PROJECT / 'third_party/AnomalyCLIP', 'R1_CLIP_CODE')
# Explicit overrides retain compatibility; otherwise prefer the reader weights directory.
_default_visual = WEIGHTS / 'ViT-L-14-336px.pt'
if not _default_visual.exists():
    _default_visual = Path.home() / '.cache/clip/ViT-L-14-336px.pt'
VISUAL = setting('FACETAD_CLIP_WEIGHTS', _default_visual, 'R1_CLIP_WEIGHTS')

def manifest_path(manifest, value):
    p = Path(value).expanduser()
    # Explicit DATA_ROOT makes all relative image/mask paths relative to that root.
    # Unset: preserve the original manifest-relative convention.
    return (p if p.is_absolute() else (DATA if 'FACETAD_DATA_ROOT' in os.environ else manifest.parent) / p).resolve()
