# PorousConvection

[![Build Status](https://github.com/arsh-k/pde-on-gpu-arsh-kumbhat/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/arsh-k/pde-on-gpu-arsh-kumbhat/actions/workflows/CI.yml?query=branch%3Amain)

In this mini-project, a stencil-based solver has been developed to simulate porous convection in 2D and 3D staggered grid domains. In the first section, we explore the use of `ParallelStencil.jl` to simulate a 2D porous convection model on xPUs (referring to both CPUs and GPUs). A similar approach is made to simulate a 3D porous convection model on xPUs in the second section. In the last section, we exploit the seamless incorporability of `ImplicitGlobalGrid.jl` with `ParallelStencil.jl` to simulate 3D porous convection using a multi-GPU configuration via `MPI.jl`. 

## Physical Model (Partial Differential Equations)

## Numerical Method (Pseudo-transient solver)

## Section 1: Porous Convection 2D

![Figure 1](./docs/porous_convection_2D_xpu_final.gif)

## Section 2: Porous Convection 3D

![Figure 2](./docs/T_3D.png)
![Figure 3](./docs/T_3D_slice_final.png)

## Section 3: Porous Convection 3D MPI

![Figure 4](./docs/T_3D_slice_mpi_final.png)
![Figure 5](./docs/porous_convection_3D_multixpu.gif)

## Conclusion