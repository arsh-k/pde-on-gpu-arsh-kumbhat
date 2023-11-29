# using Printf, LazyArrays, Plots
using Printf, Plots
using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
const USE_GPU = false
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 3, inbounds=true)
end

# CPU Array Functions

"""
    av1(A)

Returns an array computed by averaging the array `A` in all dimensions (applicable to n-dimension arrays).

#Example
```jldoctest
julia> av1([10,20,30])
2-element Vector{Float64}:
 15.0
 25.0
```
"""
@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])

"""
    avx(A)

Returns an array computed by averaging the array `A` in the first dimension (applicable to 3-dimension arrays).

#Example
```jldoctest
julia> avx([[10 20; 30 40];;;[50 60; 70 80]])
1×2×2 Array{Float64, 3}:
[:, :, 1] =
 20.0  30.0

[:, :, 2] =
 60.0  70.0
```
"""
@views avx(A) = 0.5 .* (A[1:end-1, :, :] .+ A[2:end, :, :])

"""
    avy(A)

Returns an array computed by averaging the array `A` in the second dimension (applicable to 3-dimension arrays).

#Example
```jldoctest
julia> avy([[10 20; 30 40];;;[50 60; 70 80]])
2×1×2 Array{Float64, 3}:
[:, :, 1] =
 15.0
 35.0

[:, :, 2] =
 55.0
 75.0
```
"""
@views avy(A) = 0.5 .* (A[:, 1:end-1, :] .+ A[:, 2:end, :])

"""
    avz(A)

Returns an array computed by averaging the array `A` in the third dimension (applicable to 3-dimension arrays).

#Example
```jldoctest
julia> avz([[10 20; 30 40];;;[50 60; 70 80]])
2×2×1 Array{Float64, 3}:
[:, :, 1] =
 30.0  40.0
 50.0  60.0
```
"""
@views avz(A) = 0.5 .* (A[:, :, 1:end-1] .+ A[:, :, 2:end])

# Visualizaiton function
"""
    save_array(file_name, array)

Saves an array variable `array` to the file titled `file_name.bin`.
"""
function save_array(Aname,A)
    fname = string(Aname, ".bin")
    out = open(fname, "w"); write(out, A); close(out)
end

# Hydro functions
"""
    compute_diffusion_flux!(qDx, qDy, qDz, Pf, θ_dτ_D, k_ηf, dx, dy, dz, αρg, T)

Computes the diffusive flux within a 3-dimensional staggered grid.
"""
@parallel function compute_diffusion_flux!(qDx, qDy, qDz, Pf, θ_dτ_D, k_ηf, dx, dy, dz, αρg, T)
    @inn_x(qDx) =  @inn_x(qDx) - (@inn_x(qDx) + k_ηf * (@d_xa(Pf) / dx)) / (1.0 + θ_dτ_D)
    @inn_y(qDy) =  @inn_y(qDy) - (@inn_y(qDy) + k_ηf * (@d_ya(Pf) / dy)) / (1.0 + θ_dτ_D)
    @inn_z(qDz) =  @inn_z(qDz) - (@inn_z(qDz) + k_ηf * (@d_za(Pf) / dz - αρg * @av_za(T))) / (1.0 + θ_dτ_D)
    return nothing
end

"""
    compute_Pf!(Pf, qDx, qDy, qDz, dx, dy, dz, β_dτ_D)

Computes the pressure field of a 3-dimensional staggered grid.
"""
@parallel function compute_Pf!(Pf, qDx, qDy, qDz, dx, dy, dz, β_dτ_D)
    @all(Pf)    =  @all(Pf) - (@d_xa(qDx) / dx + @d_ya(qDy) / dy + @d_za(qDz) / dz) / β_dτ_D
    return nothing
end

# Thermo functions
"""
    compute_thermal_flux!(qTx, qTy, qTz, λ_ρCp, T, dx, dy, dz, θ_dτ_T) 

Computes the heat flux within a 3-dimensional staggered grid.
"""
@parallel function compute_thermal_flux!(qTx, qTy, qTz, λ_ρCp, T, dx, dy, dz, θ_dτ_T)
    @all(qTx) = @all(qTx) - (@all(qTx) + λ_ρCp * (@d_xi(T) / dx)) / (1.0 + θ_dτ_T)
    @all(qTy) = @all(qTy) - (@all(qTy) + λ_ρCp * (@d_yi(T) / dy)) / (1.0 + θ_dτ_T)
    @all(qTz) = @all(qTz) - (@all(qTz) + λ_ρCp * (@d_zi(T) / dz)) / (1.0 + θ_dτ_T)
    return nothing
end

"""
    compute_dTdt!(dTdt, T, T_old, dt, qDx, qDy, qDz, dx, dy, dz, ϕ)

Computes the derivative of temperature with respect to the physical temporal variable within a 3-dimensional staggered grid.
"""
@parallel_indices (ix, iy, iz) function compute_dTdt!(dTdt, T, T_old, dt, qDx, qDy, qDz, dx, dy, dz, ϕ)
    nx, ny, nz = size(dTdt)
    # T, T_old - (nx, ny); qDx - (nx+1, ny); qDy - (nx, ny+1); dTdt - (nx - 2, ny - 2)
    if (ix <= nx &&  iy <= ny && iz <= nz)
        dTdt[ix, iy, iz] = (T[ix+1, iy+1, iz+1] - T_old[ix+1, iy+1, iz+1]) / dt +
        (max(qDx[ix + 1, iy + 1, iz + 1], 0.0) * (T[ix + 1, iy + 1, iz + 1] - T[ix, iy + 1, iz + 1]) / dx +
         min(qDx[ix + 2, iy + 1, iz + 1], 0.0) * (T[ix + 2, iy + 1, iz + 1] - T[ix + 1, iy + 1, iz + 1]) / dx +
         max(qDy[ix + 1, iy + 1, iz + 1], 0.0) * (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy, iz + 1]) / dy +
         min(qDy[ix + 1, iy + 2, iz + 1], 0.0) * (T[ix + 1, iy + 2, iz + 1] - T[ix + 1, iy + 1, iz + 1]) / dy +
         max(qDz[ix + 1, iy + 1, iz + 1], 0.0) * (T[ix + 1, iy + 1, iz + 1] - T[ix + 1, iy + 1, iz]) / dz +
         min(qDz[ix + 1, iy + 1, iz + 2], 0.0) * (T[ix + 1, iy + 1, iz + 2] - T[ix + 1, iy + 1, iz + 1]) / dz) / ϕ
    end
    return nothing
end

"""
    computeT!(T, dTdt, qTx, qTy, qTz, dx, dy, dz, dt, β_dτ_T)

Computes the temperature field for a 3-dimensional staggered grid.
"""
@parallel function computeT!(T, dTdt, qTx, qTy, qTz, dx, dy, dz, dt, β_dτ_T)
    @inn(T) = @inn(T) - (@all(dTdt) + @d_xa(qTx) / dx + @d_ya(qTy) / dy + @d_za(qTz) / dz) / (1.0 / dt + β_dτ_T)
    return nothing
end

# Temperature Boundary conditions
"""
    bc_yz!(T)

Sets the temperature boundary condition on the YZ plane.
"""
@parallel_indices (iy, iz) function bc_yz!(T)
    T[1  , iy, iz] = T[2    , iy, iz]
    T[end, iy, iz] = T[end-1, iy, iz]
    return
end

"""
    bc_xz!(T)

Sets the temperature boundary condition on the XZ plane.
"""
@parallel_indices (ix, iz) function bc_xz!(T)
    T[ix  , 1, iz] = T[ix    , 2, iz]
    T[ix, end, iz] = T[ix, end-1, iz]    
    return
end

# Error Check functions
"""
    compute_r_Pf!(r_Pf, qDx, qDy, qDz, dx, dy, dz)
    
Evaluates the pressure PDE residual. 
"""
@parallel function compute_r_Pf!(r_Pf, qDx, qDy, qDz, dx, dy, dz)
    @all(r_Pf) = @d_xa(qDx) / dx + @d_ya(qDy) / dy + @d_za(qDz) / dz
    return nothing
end

"""
    compute_r_T!(r_T, dTdt, qTx, qTy, qTz, dx, dy, dz)  

Evaluates the temperature PDE residual. 
"""
@parallel function compute_r_T!(r_T, dTdt, qTx, qTy, qTz, dx, dy, dz)
    @all(r_T)   = @all(dTdt) + @d_xa(qTx) / dx + @d_ya(qTy) / dy + @d_za(qTz) / dz
    return nothing
end

"""
    porous_convection_3D(;do_check, testing)

Peforms a porous convection simulation on a 3-dimensional staggered grid. The keyword arguments `do_check` and  `testing` are by default set to `false`.
"""
@views function porous_convection_3D(;do_check = false, testing = false)
    # physics
    lx, ly, lz = 40.0, 20.0, 20.0
    k_ηf       = 1.0
    # αρgx, αρgy = 0.0, 1.0
    # αρg        = sqrt(αρgx^2 + αρgy^2)
    αρg        = 1.0 #only assume to be acting in z-direction.
    ΔT         = 200.0
    ϕ          = 0.1
    Ra         = 1000
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * ly / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    # numerics
    if testing 
        nz         = 20
        ny         = nz
        nx         = 2 * (nz + 1) - 1
        nt         = 20
    else 
        nx, ny, nz = 255, 127, 127
        nt         = 2000
    end
    re_D       = 4π
    cfl        = 1.0 / sqrt(3.1)
    maxiter    = 10max(nx, ny, nz)
    ϵtol       = 1e-6
    nvis       = 50
    ncheck     = ceil(2max(nx, ny, nz))
    # preprocessing
    dx, dy, dz = lx / nx, ly / ny, lz / nz 
    xn, yn, zn = LinRange(-lx / 2, lx / 2, nx + 1), LinRange(-ly / 2, ly/ 2, ny + 1), LinRange(-lz, 0, nz + 1)
    xc, yc, zc = av1(xn), av1(yn), av1(zn)
    θ_dτ_D     = max(lx, ly, lz) / re_D / cfl / min(dx, dy, dz)
    β_dτ_D     = (re_D * k_ηf) / (cfl * min(dx, dy, dz) * max(lx, ly, lz))
    # init
    Pf           = @zeros(nx, ny, nz)
    r_Pf         = @zeros(nx, ny, nz)
    qDx, qDy, qDz= @zeros(nx + 1, ny, nz), @zeros(nx, ny + 1, nz), @zeros(nx, ny, nz + 1)
    qDx_c, qDy_c, qDz_c = zeros(nx, ny, nz), zeros(nx, ny, nz), zeros(nx, ny, nz)
    qDmag        = zeros(nx, ny, nz)
    T            = Data.Array([ΔT * exp(-xc[ix]^2 - yc[iy]^2 - (zc[iz] + lz / 2)^2) for ix = 1:nx, iy = 1:ny, iz = 1:nz])
    T[:, :, 1]  .= ΔT / 2
    T[:, :, end].= -ΔT / 2
    T_old        = Data.Array(copy(T))
    dTdt         = @zeros(nx - 2, ny - 2, nz - 2)
    r_T          = @zeros(nx - 2, ny - 2, nz - 2)
    qTx          = @zeros(nx - 1, ny - 2, nz - 2)
    qTy          = @zeros(nx - 2, ny - 1, nz - 2)
    qTz          = @zeros(nx - 2, ny - 2, nz - 1)
    # vis
    st          = ceil(Int, nx / 25)
    Xc, Yc, Zc  = [x for x in xc, y in yc, z in zc], [y for x in xc, y in yc, z in zc], [z for x in xc, y in yc, z in zc]
    Xp, Yp, Zp  = Xc[1:st:end, 1:st:end, 1:st:end], Yc[1:st:end, 1:st:end, 1:st:end], Zc[1:st:end, 1:st:end, 1:st:end]
    # action
    for it in 1:nt
        T_old .= T
        # time step
        dt = if it == 1
            0.1 * min(dx, dy, dz) / (αρg * ΔT * k_ηf)
        else
            min(5.0 * min(dx, dy, dz) / (αρg * ΔT * k_ηf), ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy)), dz / maximum(abs.(qDz))) / 3.1)
        end
        re_T   = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
        θ_dτ_T = max(lx, ly, lz) / re_T / cfl / min(dx, dy, dz)
        β_dτ_T = (re_T * λ_ρCp) / (cfl * min(dx, dy, lz) * max(lx, ly, lz))
        # iteration loop
        iter = 1
        err_D = 2ϵtol
        err_T = 2ϵtol
        while max(err_D, err_T) >= ϵtol && iter <= maxiter
            # hydro
            @parallel compute_diffusion_flux!(qDx, qDy, qDz, Pf, θ_dτ_D, k_ηf, dx, dy, dz, αρg, T)
            @parallel compute_Pf!(Pf, qDx, qDy, qDz, dx, dy, dz, β_dτ_D)
            # thermo
            @parallel compute_thermal_flux!(qTx, qTy, qTz, λ_ρCp, T, dx, dy, dz, θ_dτ_T) 
            @parallel compute_dTdt!(dTdt, T, T_old, dt, qDx, qDy, qDz, dx, dy, dz, ϕ)
            @parallel computeT!(T, dTdt, qTx, qTy, qTz, dx, dy, dz, dt, β_dτ_T)
            @parallel (1:size(T,2), 1:size(T,3)) bc_yz!(T)
            @parallel (1:size(T,1), 1:size(T,3)) bc_xz!(T)
            if iter % ncheck == 0 && do_check 
                @parallel compute_r_Pf!(r_Pf, qDx, qDy, qDz, dx, dy, dz)
                @parallel compute_r_T!(r_T, dTdt, qTx, qTy, qTz, dx, dy, dz)
                err_D = maximum(abs.(r_Pf))
                err_T = maximum(abs.(r_T))
                @printf("  iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", iter / nx, err_D, err_T)
            end
            iter += 1
        end
        @printf("it = %d, iter/nx=%.1f, err_D=%1.3e, err_T=%1.3e\n", it, iter / nx, err_D, err_T)
        # visualisation
        # if it % nvis == 0
        qDx_c .= avx(Array(qDx))
        qDy_c .= avy(Array(qDy))
        qDz_c .= avz(Array(qDz))
        qDmag .= sqrt.(qDx_c .^ 2 .+ qDy_c .^ 2 .+ qDz_c .^ 2)
        qDx_c ./= qDmag
        qDy_c ./= qDmag
        qDz_c ./= qDmag
        # qDx_p = qDx_c[1:st:end, 1:st:end,]
        # qDy_p = qDy_c[1:st:end, 1:st:end]
        # iframe = 0
        # end
    end 
    if testing == false
        p1 = heatmap(xc, zc, Array(T)[:, ceil(Int, ny / 2), :]'; xlims=(xc[1], xc[end]), ylims=(zc[1], zc[end]), aspect_ratio=1, c=:turbo)
        display(p1)
        savefig("./docs/T_3D_final.png")
        # Simulation Results
        save_array("./docs/out_T", convert.(Float32, Array(T)))
    end
    return T
end

if isinteractive()
    do_check = true
    testing  = false
    T = porous_convection_3D(;do_check, testing)
end


