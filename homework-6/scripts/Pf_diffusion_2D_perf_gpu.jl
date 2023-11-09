using Plots, Plots.Measures, Printf, LoopVectorization, CUDA, Test
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

# derivative macro initialisation
macro d_xa(A)   esc(:($A[ix+1, iy] - $A[ix, iy])) end
macro d_ya(A)   esc(:($A[ix, iy+1] - $A[ix, iy])) end

function Pf_diffusion_2D_gpu(;do_check = true, testing)
    println("-----GPU PROGRAMMING-----")
    # Pre-defining physics calculations functions
    function compute_flux_gpu!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
        ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
        iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
        if ((ix < size(Pf, 1)) && (iy <= size(Pf, 2)))
            @inbounds qDx[ix+1, iy] -= (qDx[ix+1, iy] + k_ηf_dx * (Pf[ix+1, iy] - Pf[ix, iy])) * _1_θ_dτ
        end     
        if ((ix <= size(Pf, 1)) && (iy < size(Pf, 2)))
            @inbounds qDy[ix, iy+1] -= (qDy[ix, iy+1] + k_ηf_dy * (Pf[ix, iy+1] - Pf[ix, iy])) * _1_θ_dτ     
        end
        return nothing
    end
    function update_Pf_gpu!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
        ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
        iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
        if ((ix <= size(Pf, 1)) && (iy <= size(Pf, 2)))
            @inbounds Pf[ix,iy]     -= (((qDx[ix+1, iy] - qDx[ix, iy]) * _dx) + ((qDy[ix, iy+1] - qDy[ix, iy]) * _dy)) * _β_dτ
        end
        return nothing
    end
    pow_vect = 1:8
    T_eff_vect = zeros(Float64, size(pow_vect))
    for pow in pow_vect
        # physics
        lx, ly  = 20, 20.0
        k_ηf    = 1.0
        # numerics
        ϵtol    = 1e-8
        if testing
            maxiter = 1000
            threads = (32,8)
            nx = ny = 127
            blocks  = (ceil(Int, nx ÷ threads[1]), ceil(Int, ny ÷ threads[2])) 
        else
            threads = (32,8)
            nx = ny = 32 * 2 ^ pow - 1
            blocks  = (ceil(Int, nx ÷ threads[1]), ceil(Int, ny ÷ threads[2])) 
            maxiter = max(nx, ny)
        end
        ncheck  = ceil(Int, 0.25max(nx, ny))
        cfl     = 1.0 / sqrt(2.1)
        re      = 2π
        # derived numerics
        dx, dy  = lx / nx, ly / ny
        xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
        θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
        β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
        _1_θ_dτ = 1.0 / (1.0 + θ_dτ)
        _β_dτ   = 1.0 / (β_dτ)
        _dx, _dy = 1.0 / dx, 1.0 / dy
        k_ηf_dx, k_ηf_dy = k_ηf / dx, k_ηf / dy
        # array initialisation
        Pf      = CuArray(@. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2))
        qDx     = CUDA.zeros(Float64, nx + 1, ny)
        qDy     = CUDA.zeros(Float64, nx, ny + 1)
        r_Pf    = CUDA.zeros(nx, ny)
        # iteration loop
        iter = 1; err_Pf = 2ϵtol; 
        t_tic = 0.0; niter = 0.0
        while err_Pf >= ϵtol && iter <= maxiter
            if iter == 11 #Warming up the function before you track the time.
                t_tic = Base.time(); niter = 0
            end
            # Flux computation and pressure update
            @cuda blocks=blocks threads=threads compute_flux_gpu!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ); synchronize()
            @cuda blocks=blocks threads=threads update_Pf_gpu!(Pf, qDx, qDy, _dx, _dy, _β_dτ); synchronize()
            if do_check && (iter % ncheck == 0)
                r_Pf  .= diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy
                err_Pf = maximum(abs.(r_Pf))
                @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
                display(heatmap(xc, yc, Pf'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo, clim=(0, 1)))
            end
            iter    += 1
            niter   += 1
        end
        t_toc  = Base.time() - t_tic
        size_f = sizeof(Float64)
        A_eff  = (6 * size_f * nx * ny)     # Effective main memory access per iteration [GB] 
        t_it   = t_toc/niter                # Execution time per iteration [s]
        T_eff  = A_eff/t_it                 # Effective memory throughput [GB/s]
        @printf("Iteration Loop: Time = %1.3f sec(@ %1.2f GB/s), %d iters \n", t_toc, T_eff/1e9, niter)
        T_eff_vect[pow] = T_eff/1e9
        CUDA.unsafe_free!(Pf)
        CUDA.unsafe_free!(qDx)
        CUDA.unsafe_free!(qDy)
    end
    # t_toc  = @belapsed $compute!($Pf, $qDx, $qDy, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ, $_dx_β_dτ, $_dy_β_dτ)
    # t_it   = t_toc
    # T_eff  = A_eff/t_it
    # @printf("Benchmark: Time = %1.5f sec(@ %1.2f GB/s) \n", t_toc, T_eff)
    return T_eff_vect
end

function Pf_diffusion_2D_cpu(;do_check = true, testing) #After semicolon, we add kwargs
    println("-----CPU PROGRAMMING-----")
    # Pre-defining physics calculations functions
    function compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
        nx, ny = size(Pf)
        Threads.@threads for iy = 1:ny
            # for iy=1:ny
            for ix = 1:nx-1
                @inbounds qDx[ix+1, iy] -= (qDx[ix+1, iy] + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ
            end
        end
        Threads.@threads for iy = 1:ny-1
            # for iy=1:ny-1
            for ix = 1:nx
                @inbounds qDy[ix, iy+1] -= (qDy[ix, iy+1] + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ
            end
        end
        return nothing
    end
    
    function update_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
        nx, ny = size(Pf)
        Threads.@threads for iy = 1:ny
            # for iy=1:ny
            for ix = 1:nx
                @inbounds Pf[ix, iy] -= (@d_xa(qDx) * _dx + @d_ya(qDy) * _dy) * _β_dτ
            end
        end
        return nothing
    end
    
    function compute!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _dx, _dy, _β_dτ)
        compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
        update_Pf!(Pf, qDx, qDy, _dx, _dy, _β_dτ)
        return nothing
    end
    pow_vect = 1:8
    T_eff_vect = zeros(Float64, size(pow_vect))
    for pow in pow_vect
        # physics
        lx, ly  = 20, 20.0
        k_ηf    = 1.0
        # numerics
        ϵtol    = 1e-8
        if testing
            maxiter = 1000
            nx, ny  = 127, 127
        else
            nx = ny = 32 * 2 ^ pow - 1
            maxiter = max(nx, ny)
        end
        ncheck  = ceil(Int, 0.25max(nx, ny))
        cfl     = 1.0 / sqrt(2.1)
        re      = 2π
        # derived numerics
        dx, dy  = lx / nx, ly / ny
        xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
        θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
        β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
        _1_θ_dτ = 1.0 / (1.0 + θ_dτ)
        _β_dτ   = 1.0 / (β_dτ)
        _dx, _dy = 1.0 / dx, 1.0 / dy
        k_ηf_dx, k_ηf_dy = k_ηf / dx, k_ηf / dy
        # array initialisation
        Pf       = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
        qDx, qDy = zeros(Float64, nx + 1, ny), zeros(Float64, nx, ny + 1)
        r_Pf     = zeros(nx, ny)
        # iteration loop
        iter = 1; err_Pf = 2ϵtol; 
        t_tic = 0.0; niter = 0.0
        while err_Pf >= ϵtol && iter <= maxiter
            if iter == 11 #Warming up the function before you track the time.
                t_tic = Base.time(); niter = 0
            end
            # Flux computation and pressure update
            compute!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _dx, _dy, _β_dτ)
            if do_check && (iter % ncheck == 0)
                r_Pf  .= diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy
                err_Pf = maximum(abs.(r_Pf))
                @printf("  iter/nx=%.1f, err_Pf=%1.3e\n", iter / nx, err_Pf)
                display(heatmap(xc, yc, Pf'; xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), aspect_ratio=1, c=:turbo, clim=(0, 1)))
            end
            iter    += 1
            niter   += 1
        end
        t_toc  = Base.time() - t_tic
        size_f = sizeof(Float64)
        A_eff  = (6 * size_f * nx * ny)     # Effective main memory access per iteration [GB] 
        t_it   = t_toc/niter                # Execution time per iteration [s]
        T_eff  = A_eff/t_it                 # Effective memory throughput [GB/s]
        @printf("Iteration Loop CPU: Time = %1.3f sec(@ %1.2f GB/s), %d iters \n", t_toc, T_eff/1e9, niter)
        # t_toc  = @belapsed $compute!($Pf, $qDx, $qDy, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ, $_dx_β_dτ, $_dy_β_dτ)
        # t_it   = t_toc
        # T_eff  = A_eff/t_it
        # @printf("Benchmark: Time = %1.5f sec(@ %1.2f GB/s) \n", t_toc, T_eff)
        T_eff_vect[pow] = T_eff/1e9
    end
    return T_eff_vect
end

testing = false

#T_eff_cpu = Pf_diffusion_2D_cpu(do_check = false, testing = testing) #using 4 threads for CPU computations.
T_eff_gpu         = Pf_diffusion_2D_gpu(do_check = false, testing = testing)
T_peak            = 551.618383515077 #Obtained from T_peak_evaluation.jl
nx_ = ny_         = 32 .* 2 .^ (1:8) .- 1
domain_size       = nx_ .* ny_
# Testing whether the CPU and GPU implementations produce the same value of Pf
if testing
    @testset "Diffusion - CPU & GPU Comparison" begin
        @test size(Pf_cpu) == size(Pf_gpu)
        @test all(Pf_cpu .≈ Array(Pf_gpu))
    end;
else
    plot(domain_size, T_eff_gpu, xscale=:log10, title = "Memory Throughput (GB/s) Comparison - Diffusion",
        xlabel = "Domain Size", ylabel = "Memory Throughput (GB/s)", legend=:topright, linewidth = 2,
        label = "GPU Programming")
    hline!([T_peak], label= "Peak Memory Throughput",line=(:dot, 1), linewidth = 4, linecolor = "green")
    savefig("./docs/diffusion_ex_2_task_4.png")
end

