# DiscreteCompressedSensing.jl
Software supplement for the paper  "Compressed Sensing: A Discrete Optimization Approach"  by Dimitris Bertsimas and Nicholas A. G. Johnson for which a preprint
is available [here](https://arxiv.org/pdf/2306.04647.pdf).

## Introduction

The software in this package is designed to provide high quality feasible
solutions and strong lower bounds to the regularized compressed sensing problem
given by

`min ||x||_0 + \frac{1}{\gamma} * ||x||_2^2`

`s.t. ||Ax-b||_2^2 <= \epsilon`

using algorithms described in the paper "Compressed Sensing: A Discrete
Optimization Approach"  by Dimitris Bertsimas and Nicholas A. G. Johnson.
Specifically, the above problem is reformulated exactly as a mixed integer
second order cone problem which is solved to optimality using a custom branch
and bound algorithm. The branch and bound algorithm can be halted early to
return a high quality feasible solution.

## Installation and set up
# NEED TO FINISH THIS SECTION

In order to run this software, you must install a recent version of Julia from
http://julialang.org/downloads/, a recent version of the Mosek solver (academic
licenses are freely available at
https://www.mosek.com/products/academic-licenses/), and a recent version of the
Gurobi solver (academic licenses are freely available at )


The most recent version of Julia at the time this code was last tested was Julia 1.5.1 using Mosek version 9.2.

Several packages must be installed in Julia before the code can be run.  These packages can be found in "SparseLowRankSoftware.jl". The code was last tested using the following package versions:

- Distributions v0.25.0
- JuMP v0.21.4
- LowRankApprox v0.5.0
- Mosek v1.1.3
- MosekTools v0.9.4
- SCS v0.7.1
- StatsBase v0.33.8
- TSVD v0.4.3

## Use of the SLR_AM() and SLR_BnB() functions

The three key methods in this package are CS_BnB(), perspectiveRelaxation() and
SOSRelaxation().  They both take four required  arguments: `A`, `b`, `\epsilon`, `\gamma`, as well as several optional arguments which are described in the
respective function docstring. The four required arguments correspond to the
input data to the optimization problem.

## Citing DiscreteCompressedSensing.jl

If you use DiscreteCompressedSensing.jl, we ask that you please cite the following [paper](https://arxiv.org/pdf/2306.04647.pdf):

```
@misc{bertsimas2023compressed,
      title={Compressed Sensing: A Discrete Optimization Approach},
      author={Dimitris Bertsimas and Nicholas Johnson},
      year={2023},
      eprint={2306.04647},
      archivePrefix={arXiv},
      primaryClass={eess.SP}
}
```

## Thank you

Thank you for your interest in DiscreteCompressedSensing. Please let us know if
you encounter any issues using this code, or have comments or questions.  Feel
free to email us anytime.

Dimitris Bertsimas
dbertsim@mit.edu

Nicholas A. G. Johnson
nagj@mit.edu
