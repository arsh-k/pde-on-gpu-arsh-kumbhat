using Printf, Plots
using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
const USE_GPU = false
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=true)
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=true)
end

@views av1(A) = 0.5 .* (A[1:end-1] .+ A[2:end])
@views avx(A) = 0.5 .* (A[1:end-1, :] .+ A[2:end, :])
@views avy(A) = 0.5 .* (A[:, 1:end-1] .+ A[:, 2:end])

# macro d_x(A) esc(:($A[ix+1, iy] - $A[ix, iy])) end
# macro d_y(A) esc(:($A[ix, iy+1] - $A[ix, iy])) end

# Hydro functions
@parallel function compute_diffusion_flux!(qDx, qDy, Pf, θ_dτ_D, k_ηf, dx, dy, αρgx, αρgy, T)
    @inn_x(qDx) =  @inn_x(qDx) - (@inn_x(qDx) + k_ηf * (@d_xa(Pf) / dx - αρgx * @av_xa(T))) / (1.0 + θ_dτ_D)
    @inn_y(qDy) =  @inn_y(qDy) - (@inn_y(qDy) + k_ηf * (@d_ya(Pf) / dy - αρgy * @av_ya(T))) / (1.0 + θ_dτ_D)
    return nothing
end

@parallel function compute_Pf!(Pf, qDx, qDy, dx, dy, β_dτ_D)
    @all(Pf)    =  @all(Pf) - (@d_xa(qDx) / dx + @d_ya(qDy) / dy) / β_dτ_D
    return nothing
end

# Thermo functions
@parallel function compute_thermal_flux!(qTx, qTy, λ_ρCp, T, dx, dy, θ_dτ_T)
    @all(qTx) = @all(qTx) - (@all(qTx) + λ_ρCp * (@d_xi(T) / dx)) / (1.0 + θ_dτ_T)
    @all(qTy) = @all(qTy) - (@all(qTy) + λ_ρCp * (@d_yi(T) / dy)) / (1.0 + θ_dτ_T)
    return nothing
end
# @parallel_indices (ix, iy) function compute_thermal_flux!(qTx, qTy, λ_ρCp, T, dx, dy, θ_dτ_T)
#     nx, ny = size(T)
#     if (ix <= (nx-1) && iy <= (ny-2)) 
#         qTx[ix, iy] -= (qTx[ix, iy] + λ_ρCp * (@d_x(T[ix, iy + 1]) / dx)) / (1.0 + θ_dτ_T)
#     end
#     if (ix <= (nx-2) && iy <= (ny-1))
#         qTy[ix, iy] -= (qTy[ix, iy] + λ_ρCp * (@d_y(T[ix + 1, iy]) / dy)) / (1.0 + θ_dτ_T)
#     end
#     return nothing
# end

@parallel_indices (ix, iy) function compute_dTdt!(dTdt, T, T_old, dt, qDx, qDy, dx, dy, ϕ)
    nx, ny = size(dTdt)
    # T, T_old - (nx, ny); qDx - (nx+1, ny); qDy - (nx, ny+1); dTdt - (nx - 2, ny - 2)
    if (ix <= nx &&  iy <= ny)
        dTdt[ix, iy] = (T[ix+1, iy+1] - T_old[ix+1, iy+1]) / dt +
        (max(qDx[ix + 1, iy + 1], 0.0) * (T[ix + 1, iy + 1] - T[ix, iy + 1]) / dx +
         min(qDx[ix + 2, iy + 1], 0.0) * (T[ix + 2, iy + 1] - T[ix + 1, iy + 1]) / dx +
         max(qDy[ix + 1, iy + 1], 0.0) * (T[ix + 1, iy + 1] - T[ix + 1, iy]) / dy +
         min(qDy[ix + 1, iy + 2], 0.0) * (T[ix + 1, iy + 2] - T[ix + 1, iy + 1]) / dy) / ϕ
    end
    return nothing
end

@parallel function computeT!(T, dTdt, qTx, qTy, dx, dy, dt, β_dτ_T)
    @inn(T) = @inn(T) - (@all(dTdt) + @d_xa(qTx) / dx + @d_ya(qTy) / dy) / (1.0 / dt + β_dτ_T)
    return nothing
end

# Temperature Boundary condition
@parallel_indices (iy) function bc_x!(T)
    T[1  , iy] = T[2    , iy]
    T[end, iy] = T[end-1, iy]
    return
end

# Error Check functions
@parallel function compute_r_Pf!(r_Pf, qDx, qDy, dx, dy)
    @all(r_Pf) = @d_xa(qDx) / dx + @d_ya(qDy) / dy
    return nothing
end

@parallel function compute_r_T!(r_T, dTdt, qTx, qTy, dx, dy)
    @all(r_T)   = @all(dTdt) + @d_xa(qTx) / dx + @d_ya(qTy) / dy
    return nothing
end

@views function porous_convection_2D(;do_check = false, testing = false)
    # physics
    lx, ly     = 40.0, 20.0
    k_ηf       = 1.0
    αρgx, αρgy = 0.0, 1.0
    αρg        = sqrt(αρgx^2 + αρgy^2)
    ΔT         = 200.0
    ϕ          = 0.1
    Ra         = 1000
    λ_ρCp      = 1 / Ra * (αρg * k_ηf * ΔT * ly / ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    # numerics
    if testing
        ny           = 20
        nx           = 2 * (ny + 1) - 1
        nt           = 20
    else
        nx, ny     = 1023, 511
        nt         = 4000
    end
    re_D       = 4π
    cfl        = 1.0 / sqrt(2.1)
    maxiter    = 10max(nx, ny)
    ϵtol       = 1e-6
    nvis       = 50
    ncheck     = ceil(2max(nx, ny))
    # preprocessing
    dx, dy = lx / nx, ly / ny
    xn, yn = LinRange(-lx / 2, lx / 2, nx + 1), LinRange(-ly, 0, ny + 1)
    xc, yc = av1(xn), av1(yn)
    θ_dτ_D = max(lx, ly) / re_D / cfl / min(dx, dy)
    β_dτ_D = (re_D * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    # init
    Pf           = @zeros(nx, ny)
    r_Pf         = @zeros(nx, ny)
    qDx, qDy     = @zeros(nx + 1, ny), @zeros(nx, ny + 1)
    qDx_c, qDy_c = zeros(nx, ny), zeros(nx, ny)
    qDmag        = zeros(nx, ny)
    T            = Data.Array(@. ΔT * exp(-xc^2 - (yc' + ly / 2)^2))
    T[:, 1]      .= ΔT / 2
    T[:, end]    .= -ΔT / 2
    T_old        = Data.Array(copy(T))
    dTdt         = @zeros(nx - 2, ny - 2)
    r_T          = @zeros(nx - 2, ny - 2)
    qTx          = @zeros(nx - 1, ny - 2)
    qTy          = @zeros(nx - 2, ny - 1)
    # vis
    st     = ceil(Int, nx / 25)
    Xc, Yc = [x for x in xc, y in yc], [y for x in xc, y in yc]
    Xp, Yp = Xc[1:st:end, 1:st:end], Yc[1:st:end, 1:st:end]
    # action (animation is not added for testing)
    # anim = @animate for it in 1:nt
    for it in 1:nt
        T_old .= T
        # time step
        dt = if it == 1
            0.1 * min(dx, dy) / (αρg * ΔT * k_ηf)
        else
            min(5.0 * min(dx, dy) / (αρg * ΔT * k_ηf), ϕ * min(dx / maximum(abs.(qDx)), dy / maximum(abs.(qDy))) / 2.1)
        end
        re_T   = π + sqrt(π^2 + ly^2 / λ_ρCp / dt)
        θ_dτ_T = max(lx, ly) / re_T / cfl / min(dx, dy)
        β_dτ_T = (re_T * λ_ρCp) / (cfl * min(dx, dy) * max(lx, ly))
        # iteration loop
        iter = 1
        err_D = 2ϵtol
        err_T = 2ϵtol
        while max(err_D, err_T) >= ϵtol && iter <= maxiter
            # hydro
            @parallel compute_diffusion_flux!(qDx, qDy, Pf, θ_dτ_D, k_ηf, dx, dy, αρgx, αρgy, T)
            @parallel compute_Pf!(Pf, qDx, qDy, dx, dy, β_dτ_D)
            # thermo
            @parallel compute_thermal_flux!(qTx, qTy, λ_ρCp, T, dx, dy, θ_dτ_T) 
            @parallel compute_dTdt!(dTdt, T, T_old, dt, qDx, qDy, dx, dy, ϕ)
            @parallel computeT!(T, dTdt, qTx, qTy, dx, dy, dt, β_dτ_T)
            @parallel (1:size(T,2)) bc_x!(T)
            if iter % ncheck == 0 && do_check 
                @parallel compute_r_Pf!(r_Pf, qDx, qDy, dx, dy)
                @parallel compute_r_T!(r_T, dTdt, qTx, qTy, dx, dy)
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
        qDmag .= sqrt.(qDx_c .^ 2 .+ qDy_c .^ 2)
        qDx_c ./= qDmag
        qDy_c ./= qDmag
        qDx_p = qDx_c[1:st:end, 1:st:end]
        qDy_p = qDy_c[1:st:end, 1:st:end]
        if testing == false
            heatmap(xc, yc, Array(T)'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo)
            quiver!(Xp[:], Yp[:]; quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black)
        end
    # end every nvis
    end
    # gif(anim, "./docs/porous_convection_2D_xpu_final.gif"; fps = 10)
    return T
end

if isinteractive()
    do_check = true
    testing  = false
    T = porous_convection_2D(;do_check, testing)
end
