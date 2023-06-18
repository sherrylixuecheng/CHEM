# CHEM

<p align="center">
  <a href="https://github.com/sherrylixuecheng/CHEM">
    <img width=80% src="https://github.com/sherrylixuecheng/CHEM/blob/main/diagram.png">
  </a>
</p>


## Overview
This repository includes the codes and results for the manuscript:
***Towards chemical accuracy with shallow quantum circuits: A Clifford Hamiltonian engineering approach***

## Installation and usage
This repository requires to install two open-sourced packages: 

* [TenCirChem](https://github.com/tencent-quantum-lab/tencirchem): ```pip install tencirchem``` 


## Content list

### Files

* [clifford_autotransform.py](clifford_autotransform.py): This is the codes that implement and run the CHEM approach for seven tested electronic structure systems with different circuit depth and different initializations.

* [scipy_opt.py](scipy_opt.py): This is the python file to generate the results default HEA circuit without CHEM. 

* [plot.ipynb](plot.ipynb): This is a a jupyter notebook to generate all the plots used in the paper.

* [stats.xlsx](stats.xlsx): This is a supplemental excel to summarize the optimized energies and optimized iterations in L-BFGS-B optimization with or without CHEM for different tesed systems.


### Folders

* [results](results): each subfolder contains the collected results for the corresponding. 
- [clifford_autotransform](results/clifford_autotransform): to collect the results with CHEM method
- [clifford_init](results/clifford_init): to collect the results with non-transformed Clifford.
- [random_init](results/random_init): to collect the results without CHEM and directly using random initial guesses to optimize HEA.
- Several ``.npy`` files summarizes the stats of the corresponding stats

* [figures](figures): contains the figures generated by [plot.ipynb](plot.ipynb) in this folder.

## Please cite us as

```
@article{sun2023chem,
  title={Towards chemical accuracy with shallow quantum circuits: A Clifford Hamiltonian engineering approach},
  author={Sun, Jiace and Cheng, Lixue and Li, Weitang},
  journal={arXiv preprint arXiv:230x.xxxx},
  year={2023}
}
```