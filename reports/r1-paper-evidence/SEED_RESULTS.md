# Frozen members and probability ensembles

Class-macro percentages. Core averages 11 eligible datasets per task. Member dispersion uses sample SD (n=3). The ensemble is scored once after averaging predictions.

|Dataset|Model|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO|
|---|---|---:|---:|---:|---:|---:|
| mvtec | 111 | 92.803 | 96.669 | 91.502 | 40.428 | 87.221 |
| mvtec | 222 | 94.251 | 97.482 | 91.461 | 39.151 | 87.194 |
| mvtec | 333 | 93.308 | 96.929 | 91.514 | 40.116 | 87.392 |
| mvtec | member mean | 93.454 ± 0.735 | 97.027 ± 0.416 | 91.492 ± 0.028 | 39.898 ± 0.666 | 87.269 ± 0.108 |
| mvtec | static A ensemble | 93.693 | 97.238 | 90.975 | 34.339 | 85.389 |
| mvtec | R1 ensemble | 93.897 | 97.216 | 91.513 | 40.118 | 87.388 |
| mvtec | R1 ensemble minus member mean | 0.443 | 0.189 | 0.021 | 0.219 | 0.119 |
| visa | 111 | 83.847 | 86.604 | 95.332 | 25.148 | 88.887 |
| visa | 222 | 84.037 | 87.097 | 95.393 | 24.993 | 88.853 |
| visa | 333 | 82.685 | 85.998 | 95.386 | 24.892 | 88.328 |
| visa | member mean | 83.523 ± 0.732 | 86.566 ± 0.550 | 95.370 ± 0.033 | 25.011 ± 0.129 | 88.689 ± 0.314 |
| visa | static A ensemble | 83.806 | 86.572 | 95.411 | 21.192 | 87.509 |
| visa | R1 ensemble | 83.747 | 86.714 | 95.381 | 25.079 | 88.764 |
| visa | R1 ensemble minus member mean | 0.223 | 0.148 | 0.010 | 0.068 | 0.075 |
| core | 111 | 85.800 | 88.748 | 93.775 | 46.962 | 81.933 |
| core | 222 | 86.038 | 88.772 | 93.765 | 46.390 | 82.173 |
| core | 333 | 86.201 | 88.900 | 93.716 | 46.298 | 81.726 |
| core | member mean | 86.013 ± 0.201 | 88.807 ± 0.081 | 93.752 ± 0.031 | 46.550 ± 0.360 | 81.944 ± 0.224 |
| core | static A ensemble | 85.221 | 88.291 | 92.474 | 42.674 | 79.743 |
| core | R1 ensemble | 86.300 | 89.062 | 93.781 | 46.709 | 82.001 |
| core | R1 ensemble minus member mean | 0.287 | 0.255 | 0.029 | 0.159 | 0.058 |
