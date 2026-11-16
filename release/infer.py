"""Write an image probability and a 518×518 probability map per input."""

import argparse

import numpy as np
from PIL import Image
from runtime import Ensemble

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("visa", "mvtec"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("images", nargs="+")
    args = p.parse_args()
    model = Ensemble(args.source)
    probabilities = []
    maps = []
    for path in args.images:
        with Image.open(path) as im:
            prob, heat = model.predict([im])
        probabilities.append(prob[0])
        maps.append(heat.cpu().numpy().reshape(518, 518))
    np.savez_compressed(args.output, files=args.images, image=probabilities, pixel=maps)
