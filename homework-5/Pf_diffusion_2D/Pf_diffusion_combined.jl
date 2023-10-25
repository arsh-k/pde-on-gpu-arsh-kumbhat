using Plots, Plots.Measures, Printf, BenchmarkTools, LoopVectorization
default(size=(800, 700), framestyle=:box, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

function Pf_diffusion_2D_Teff() 
    function Pf_diffusion_Teff!(qDx, qDy, Pf, k_ηf, θ_dτ, β_dτ, dx, dy)
        qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ k_ηf .* (diff(Pf, dims=1) ./ dx)) ./ (1.0 + θ_dτ) #3 memory accesses
        qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ k_ηf .* (diff(Pf, dims=2) ./ dy)) ./ (1.0 + θ_dτ) #3
        Pf              .-= (diff(qDx, dims=1) ./ dx .+ diff(qDy, dims=2) ./ dy) ./ β_dτ          #4 - reading Pf, qDx, qDy and writing to Pf
    end
    println("----------ARRAY PROGRAMMING-----------")
    pow_vect = 1:8
    T_eff_vect  = zeros(Float64, size(pow_vect)) 
    for pow in pow_vect  
        # physics
        lx, ly   = 20, 20.0
        k_ηf     = 1.0
        # numerics
        nx = ny  = 16 * 2 ^ pow
        cfl      = 1.0 / sqrt(2.1)
        re       = 2π
        # derived numerics
        dx, dy   = lx / nx, ly / ny
        xc, yc   = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
        θ_dτ     = max(lx, ly) / re / cfl / min(dx, dy)
        β_dτ     = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
        # array initialisation
        Pf       = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
        qDx      = zeros(Float64, nx + 1, ny)
        qDy      = zeros(Float64, nx, ny + 1)
        t_it     = @belapsed $Pf_diffusion_Teff!($qDx, $qDy, $Pf, $k_ηf, $θ_dτ, $β_dτ, $dx, $dy)   # Execution time per iteration [s]
        size_f   = sizeof(Float64)
        A_eff    = (8 * size_f * nx * ny) + (size_f * (nx+1) * ny) + (size_f * nx * (ny+1))      # Effective main memory access per iteration [GB]                                                                    
        T_eff    = A_eff/t_it                                                                    # Effective memory throughput [GB/s]
        println("Domain Size = $(nx*ny)")                                                                   
        @printf("Time = %1.5f sec(@ %1.2f GB/s) \n", t_it, T_eff / 1e9)
        T_eff_vect[pow] = T_eff/1e9
    end
    return T_eff_vect
end

function Pf_diffusion_2D_perf() 
    function Pf_diffusion_perf(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _dx_β_dτ, _dy_β_dτ)
        qDx[2:end-1, :] .-= (qDx[2:end-1, :] .+ k_ηf_dx .* (diff(Pf, dims=1))) .* _1_θ_dτ               #3 memory accesses
        qDy[:, 2:end-1] .-= (qDy[:, 2:end-1] .+ k_ηf_dy .* (diff(Pf, dims=2))) .* _1_θ_dτ               #3
        Pf              .-= (diff(qDx, dims=1) .* _dx_β_dτ .+ diff(qDy, dims=2) .* _dy_β_dτ)            #4    
    end
    println("-----------ARRAY PROGRAMMING (INVERSE MULTIPLICATION)----------")
    pow_vect = 1:8
    T_eff_vect  = zeros(Float64, size(pow_vect)) 
    for pow in pow_vect
        # physics
        lx, ly  = 20, 20.0
        k_ηf    = 1.0
        # numerics
        nx = ny = 16 * 2 ^ pow
        cfl     = 1.0 / sqrt(2.1)
        re      = 2π
        # derived numerics
        dx, dy  = lx / nx, ly / ny
        xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
        θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
        β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
        k_ηf_dx, k_ηf_dy = k_ηf/dx , k_ηf/dy
        _dx_β_dτ, _dy_β_dτ = 1/dx/β_dτ, 1/dy/β_dτ
        _1_θ_dτ = 1.0./(1.0 + θ_dτ) 
        # array initialisation
        Pf      = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
        qDx     = zeros(Float64, nx + 1, ny)
        qDy     = zeros(Float64, nx, ny + 1)
        # memory throughput evaluation
        t_it    = @belapsed  $Pf_diffusion_perf($qDx, $qDy, $Pf, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ, $_dx_β_dτ, $_dy_β_dτ)                                                          # Execution time per iteration [s]
        size_f  = sizeof(Float64)
        A_eff   = (8 * size_f * nx * ny) + (size_f * (nx+1) * ny) + (size_f * nx * (ny+1))      # Effective main memory access per iteration [GB]                                                                  
        T_eff   = A_eff/t_it                                                                    # Effective memory throughput [GB/s]
        println("Domain Size = $(nx*ny)")  
        @printf("Time = %1.5f sec(@ %1.2f GB/s) \n", t_it, T_eff/1e9)
        T_eff_vect[pow] = T_eff/1e9
    end
    return T_eff_vect
end

# derivative macro initialisation
macro d_xa(A)   esc(:($A[ix+1, iy] - $A[ix, iy])) end
macro d_ya(A)   esc(:($A[ix, iy+1] - $A[ix, iy])) end

function Pf_diffusion_2D_perf_loop() 
    function Pf_diffusion_perf_loop(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _dx_β_dτ, _dy_β_dτ)
        nx, ny = size(Pf)
        for iy=1:ny
            for ix=1:nx-1
                @inbounds qDx[ix+1, iy] -= (qDx[ix+1, iy] + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ     
            end
        end
        for iy=1:ny-1
            for ix=1:nx
                @inbounds qDy[ix, iy+1] -= (qDy[ix, iy+1] + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ     
            end
        end
        for iy=1:ny
            for ix=1:nx
                @inbounds Pf[ix,iy]     -= ((@d_xa(qDx) * _dx_β_dτ) + (@d_ya(qDy) * _dy_β_dτ))
            end                    
        end
    end
    println("-----------KERNEL (WITHOUT THREADS)----------")
    pow_vect = 1:8
    T_eff_vect  = zeros(Float64, size(pow_vect))
    for pow in pow_vect
        # physics
        lx, ly  = 20, 20.0
        k_ηf    = 1.0
        # numerics
        nx = ny = 16 * 2 ^ pow 
        cfl     = 1.0 / sqrt(2.1)
        re      = 2π
        # derived numerics
        dx, dy  = lx / nx, ly / ny
        xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
        θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
        β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
        k_ηf_dx, k_ηf_dy = k_ηf/dx , k_ηf/dy
        _dx_β_dτ, _dy_β_dτ = 1/dx/β_dτ, 1/dy/β_dτ
        _1_θ_dτ = 1.0./(1.0 + θ_dτ) 
        # array initialisation
        Pf      = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
        qDx     = zeros(Float64, nx + 1, ny)
        qDy     = zeros(Float64, nx, ny + 1)
        t_it    = @belapsed $Pf_diffusion_perf_loop($qDx, $qDy, $Pf, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ, $_dx_β_dτ, $_dy_β_dτ) # Execution time per iteration [s]
        size_f  = sizeof(Float64)
        A_eff   = (8 * size_f * nx * ny) + (size_f * (nx+1) * ny) + (size_f * nx * (ny+1))      # Effective main memory access per iteration [GB] 
        T_eff   = A_eff/t_it                                                                    # Effective memory throughput [GB/s]
        @printf("Time = %1.5f sec(@ %1.2f GB/s)\n", t_it, T_eff/1e9)
        T_eff_vect[pow] = T_eff/1e9
    end
    return T_eff_vect
end

function Pf_diffusion_2D_perf_loop_fun()
    # Pre-defining physics calculations functions
    function compute_flux!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
        nx, ny = size(Pf)
        Threads.@threads for iy=1:ny
            for ix=1:nx-1
                @inbounds qDx[ix+1, iy] -= (qDx[ix+1, iy] .+ k_ηf_dx .* @d_xa(Pf)) * _1_θ_dτ     
            end
        end
        Threads.@threads for iy=1:ny-1
            for ix=1:nx
                @inbounds qDy[ix, iy+1] -= (qDy[ix, iy+1] .+ k_ηf_dy .* @d_ya(Pf)) * _1_θ_dτ     
            end
        end
        return nothing
    end
    function update_Pf!(Pf, qDx, qDy, _dx_β_dτ, _dy_β_dτ)
        nx, ny = size(Pf)
        Threads.@threads for iy=1:ny
            for ix=1:nx
                @inbounds Pf[ix,iy]     -= ((@d_xa(qDx) * _dx_β_dτ) + (@d_ya(qDy) * _dy_β_dτ))
            end                    
        end
        return nothing
    end
    # Combining the physics functions into a single compile function
    function compute!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _dx_β_dτ, _dy_β_dτ)
        compute_flux!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
        update_Pf!(Pf, qDx, qDy, _dx_β_dτ, _dy_β_dτ)
        return nothing
    end
    println("-----------KERNEL (FOUR THREADS)----------")
    pow_vect = 1:8
    T_eff_vect  = zeros(Float64, size(pow_vect))
    for pow in pow_vect
        # physics
        lx, ly = 20, 20.0
        k_ηf   = 1.0
        # numerics
        nx = ny = 16 * 2 ^ pow 
        cfl     = 1.0 / sqrt(2.1)
        re      = 2π
        # derived numerics
        dx, dy  = lx / nx, ly / ny
        xc, yc  = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)
        θ_dτ    = max(lx, ly) / re / cfl / min(dx, dy)
        β_dτ    = (re * k_ηf) / (cfl * min(dx, dy) * max(lx, ly))
        k_ηf_dx, k_ηf_dy    = k_ηf/dx , k_ηf/dy
        _dx_β_dτ, _dy_β_dτ  = 1/dx/β_dτ, 1/dy/β_dτ
        _1_θ_dτ = 1.0./(1.0 + θ_dτ) 
        # array initialisation
        Pf      = @. exp(-(xc - lx / 2)^2 - (yc' - ly / 2)^2)
        qDx     = zeros(Float64, nx + 1, ny)
        qDy     = zeros(Float64, nx, ny + 1)
        t_it    = @belapsed $compute!($Pf, $qDx, $qDy, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ, $_dx_β_dτ, $_dy_β_dτ)
        size_f  = sizeof(Float64)
        A_eff   = (8 * size_f * nx * ny) + (size_f * (nx+1) * ny) + (size_f * nx * (ny+1))
        T_eff   = A_eff/t_it
        @printf("Benchmark: Time = %1.5f sec(@ %1.2f GB/s) \n", t_it, T_eff/ 1e9)
        T_eff_vect[pow] = T_eff/1e9
    end
    return T_eff_vect
end

# Plotting Memory Throughput vs Domain Size
T_eff_init              = Pf_diffusion_2D_Teff()
T_eff_perf              = Pf_diffusion_2D_perf()
T_eff_perf_loop         = Pf_diffusion_2D_perf_loop()
T_eff_perf_loop_fun     = Pf_diffusion_2D_perf_loop_fun()
nx_ = ny_               = 16 * 2 .^ (1:8)
domain_size             = nx_ .* ny_
max_bandwidth           = 37.5
max_memory_throughput   = maximum([T_eff_init; T_eff_perf; T_eff_perf_loop; T_eff_perf_loop_fun])
plot(domain_size, [T_eff_init, T_eff_perf, T_eff_perf_loop, T_eff_perf_loop_fun], xscale=:log10, title = "Memory Throughput (GB/s) Comparison - Diffusion",
        xlabel = "Domain Size", ylabel = "Memory Throughput (GB/s)", legend=:topright, linewidth = 2,
        label = ["Array Programming" "Array Prog (Inverse Multiplication)" "Kernel Programming" "Kernel Programming (4 Threads)"])
hline!([max_bandwidth], label= "Maximum CPU Bandwidth",line=(:dot, 1), linewidth = 4, linecolor = "green")
hline!([max_memory_throughput], label = "Maximum Memory Throughput - $(round(max_memory_throughput, sigdigits=4))", 
        line=(:dash, 1), linewidth = 4, linecolor = "red")
savefig("../docs/diffusion_ex_2_task_3.png")
