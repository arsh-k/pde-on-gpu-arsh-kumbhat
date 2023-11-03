using Plots, Plots.Measures, Printf
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

function Pf_diffusion_2D(;do_check = true) #After semicolon, we add kwargs
    # physics
    lx, ly  = 20, 20.0
    k_ηf    = 1.0
    # numerics
    nx, ny  = 511, 511
    ϵtol    = 1e-8
    maxiter = max(nx, ny)
    ncheck  = ceil(Int, 0.25max(nx, ny))
    cfl     = 1.0 / sqrt(2.1)
    re      = 2π
    # derived numerics
    dx, dy  = lx / nx, ly / ny
    xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
    θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
    β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
    # array initialisation
    Pf      = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
    qDx     = zeros(Float64, nx + 1, ny)
    qDy     = zeros(Float64, nx, ny + 1)
    r_Pf    = zeros(nx, ny)
    # iteration loop
    iter = 1; err_Pf = 2ϵtol; 
    t_tic = 0.0; niter = 0.0
    while err_Pf >= ϵtol && iter <= maxiter
        if iter == 11 #Warming up the function before you track the time.
            t_tic = Base.time(); niter = 0
        end
        qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ k_ηf .* (diff(Pf, dims=1) ./ dx)) ./ (1.0 + θ_dτ) #3 memory accesses
        qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ k_ηf .* (diff(Pf, dims=2) ./ dy)) ./ (1.0 + θ_dτ) #3
        Pf              .-= (diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy) ./ β_dτ          #4 - reading Pf, qDx, qDy and writing to Pf
        if do_check && (iter % ncheck == 0)
            r_Pf .= diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy
            err_Pf = maximum(abs.(r_Pf))
            @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
            display(heatmap(xc, yc, Pf'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo, clim=(0, 1)))
        end
        iter    += 1
        niter   += 1
    end
    t_toc  = Base.time() - t_tic
    size_f = sizeof(Float64)
    A_eff  = (8 * size_f * nx * ny) + (size_f * (nx+1) * ny) + (size_f * nx * (ny+1))      # Effective main memory access per iteration [GB] 
    t_it   = t_toc/niter                                                                   # Execution time per iteration [s]
    T_eff  = A_eff/t_it                                                                    # Effective memory throughput [GB/s]
    # @show t_toc
    @printf("Time = %1.3f sec(@ %1.2f GB/s), %d iters \n", t_toc, T_eff, niter)
    return
end

Pf_diffusion_2D(do_check = false)
