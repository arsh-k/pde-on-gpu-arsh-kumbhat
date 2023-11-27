# PorousConvection

[![Build Status](https://github.com/arsh-k/pde-on-gpu-arsh-kumbhat/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/arsh-k/pde-on-gpu-arsh-kumbhat/actions/workflows/CI.yml?query=branch%3Amain)

In this mini-project, a stencil-based solver has been developed to simulate porous convection in 2D and 3D staggered grid domains. In the first section, we explore the use of [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) to simulate a 2D porous convection model on xPUs (referring to both CPUs and GPUs). A similar approach is made to simulate a 3D porous convection model on xPUs in the second section. In the last section, we exploit the seamless incorporability of [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) with [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) to simulate 3D porous convection using a multi-GPU configuration via [MPI.jl](https://github.com/JuliaParallel/MPI.jl). The bonus section involves the use of a popular automatic documentation tool [Literate.jl](https://github.com/fredrikekre/Literate.jl).

[ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl) ensures that a single Julia script allows for the implementation of a stencil-based solver on both CPUs and GPUs. Its seamless incorporability with [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) becomes a powerful tool for multi-XPU implementations. The reason being that [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) condenses necessary [MPI.jl](https://github.com/JuliaParallel/MPI.jl) code blocks into simple functions such as `update_halo!` (updating local domain boundary conditions between separate processes) and `gather!` (particularly used as a visualization tool in this project, gathers the arrays from the local processes to form a single global array). [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl) also provides us with a `@hide_communication` macro which helps us hide communication between the processes behind computation which helps reduce the total execution time of our simulation on multiple processes.

## Physical Model (Partial Differential Equations)

##### Fluid Flow in Porous Media

In our physical model, the fluid is considered to be incompressible (i.e., $\rho$ remains constant) and hence we obtain the following conservation of mass equation.

$$
\nabla \cdot(\phi \boldsymbol{v})=0
$$

Here, $\phi$ is the porosity which remains constant if our porous material is undeformable. In our numerical solution, we assume the porosity to be constant.

##### Darcy's Law

We define a quantity $\boldsymbol{q}_{\boldsymbol{D}} = \phi \boldsymbol{v}$ called the Darcy flux or Darcy velocity.  

$$
\boldsymbol{q}_{\boldsymbol{D}}=-\frac{k}{\eta}(\nabla p-\rho \boldsymbol{g})
$$

##### Pressure Residual 

We obtain the pressure residual by substituting the Darcy flux into the mass conservation equation for an incompressible fluid. We get the following equation:

$$
\nabla \cdot\left[\frac{k}{\eta}(\nabla p-\rho \boldsymbol{g})\right]=0
$$

##### Heat Convection in Porous Media

The following equation represents the energy conservation equation for the fluid in porous media:

$$
\rho c_p \frac{\partial T}{\partial t}+\rho c_p \boldsymbol{v} \cdot \nabla T+\nabla \cdot \boldsymbol{q}_{\boldsymbol{F}}=0
$$

where $c_{p}$ is the specific heat capacity of the fluid, $\boldsymbol{q}_{\boldsymbol{F}}$ is the conductive heat flux and $t$ is the physical time.

Analogous to Darcy's law, there exists the Fourier's law which relates the conductive heat flux to the temperature gradient:

$$
\boldsymbol{q}_{\boldsymbol{F}}=-\lambda \nabla T
$$

where $\lambda$ is the thermal conductivity (which is assumed constant in our simulation). By substituting the Darcy flux equation and the 

$$
$$

##### Boussinesq Approximation
$$

$$

## Numerical Method (Pseudo-transient solver)

## Section 1: Porous Convection 2D

![Figure 1](./docs/porous_convection_2D_xpu_final.gif)

## Section 2: Porous Convection 3D

![Figure 2](./docs/T_3D.png)
![Figure 3](./docs/T_3D_slice_final.png)

## Section 3: Porous Convection 3D MPI

![Figure 4](./docs/T_3D_slice_mpi_final.png)
![Figure 5](./docs/porous_convection_3D_multixpu.gif)

## Bonus Section: Documentation 

## Conclusion