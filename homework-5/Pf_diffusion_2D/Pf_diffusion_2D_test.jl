using Plots, Plots.Measures, Printf, BenchmarkTools, LoopVectorization, Test
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

# derivative macro initialisation
macro d_xa(A)   esc(:($A[ix+1, iy] - $A[ix, iy])) end
macro d_ya(A)   esc(:($A[ix, iy+1] - $A[ix, iy])) end

function Pf_diffusion_2D(;do_check)
    # Pre-defining physics calculations functions
    function compute_flux!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
        nx, ny = size(Pf)
        @tturbo for iy=1:ny
            for ix=1:nx-1
                qDx[ix+1, iy] -= (qDx[ix+1, iy] + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ     
            end
        end
        @tturbo for iy=1:ny-1
            for ix=1:nx
                qDy[ix, iy+1] -= (qDy[ix, iy+1] + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ     
            end
        end
        return nothing
    end
    function update_Pf!(Pf, qDx, qDy, _dx_β_dτ, _dy_β_dτ)
        nx, ny = size(Pf)
        @tturbo for iy=1:ny
            for ix=1:nx
                Pf[ix,iy]     -= ((@d_xa(qDx) * _dx_β_dτ) + (@d_ya(qDy) * _dy_β_dτ))
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
    # Looping through all possible nx, ny domain sizes     
    pow_vect  = 2:5
    Pf_actual = zeros(Float64, 4, 3)
    for pow in pow_vect
        # physics
        lx, ly  = 20, 20.0
        k_ηf    = 1.0
        # numerics
        nx = ny = (16 * (2 ^ pow)) - 1
        ϵtol    = 1e-8
        maxiter = 500
        ncheck  = ceil(Int, 0.25max(nx, ny))
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
        r_Pf    = zeros(nx, ny)   
        # iteration loop
        iter = 1; err_Pf = 2ϵtol; 
        t_tic = 0.0; niter = 0.0
        while err_Pf >= ϵtol && iter <= maxiter
            if iter == 11 #Warming up the function before you track the time.
                t_tic = Base.time(); niter = 0
            end
            # Flux computation and pressure update
            compute!(Pf, qDx, qDy, k_ηf_dx, k_ηf_dy, _1_θ_dτ, _dx_β_dτ, _dy_β_dτ)
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
        A_eff  = (10 * size_f * nx * ny)                                                       # Effective main memory access per iteration [GB] 
        t_it   = t_toc/niter                                                                   # Execution time per iteration [s]
        T_eff  = A_eff/t_it                                                                    # Effective memory throughput [GB/s]
        # @printf("Iteration Loop: Time = %1.3f sec(@ %1.2f GB/s), %d iters \n", t_toc, T_eff, niter)
        # t_toc = @belapsed $compute!($Pf, $qDx, $qDy, $k_ηf_dx, $k_ηf_dy, $_1_θ_dτ, $_dx_β_dτ, $_dy_β_dτ)
        # t_it  = t_toc
        # T_eff = A_eff/t_it
        # @printf("Benchmark: Time = %1.5f sec(@ %1.2f GB/s) \n", t_toc, T_eff)
        xtest = [5, Int(cld(0.6*lx, dx)), nx-10]
        ytest = Int(cld(0.5*ly, dy))
        Pf_actual[pow-1, :] = Pf[xtest, ytest]
    end
    # Testing Diffusion Solver
    @testset "Diffusion Solver Test" begin
        @test all(Pf_actual[1, :] .≈ [0.00785398056115133, 0.007853980637555755, 0.007853978592411982])
        @test all(Pf_actual[2, :] .≈ [0.00787296974549236, 0.007849556884184108, 0.007847181374079883])
        @test all(Pf_actual[3, :] .≈ [0.00740912103848251, 0.009143711648167267, 0.007419533048751209])
        @test all(Pf_actual[4, :] .≈ [0.00566813765849919, 0.004348785338575644, 0.005618691590498087])
    end;
    return
end

Pf_diffusion_2D(do_check = false)