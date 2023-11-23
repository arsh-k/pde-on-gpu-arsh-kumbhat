# juliap -O3 diffusion_2D_perf_xpu.jl
using MPI
MPI.Init()
const USE_GPU = true
using ParallelStencil, ImplicitGlobalGrid
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2, inbounds=true)
    file_name = "gpu"
else
    @init_parallel_stencil(Threads, Float64, 2, inbounds=true)
    file_name = "cpu"
end
using Plots, Printf, MAT, Plots.Measures
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)
ENV["GKSwstype"]="nul";

if !@isdefined do_save; do_save = true end

# macros to avoid array allocation
macro qx(ix, iy) esc(:(-D_dx * (C[$ix+1, $iy+1] - C[$ix, $iy+1]))) end
macro qy(ix, iy) esc(:(-D_dy * (C[$ix+1, $iy+1] - C[$ix+1, $iy]))) end

@parallel_indices (ix, iy) function compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size_C1_2, size_C2_2)
    if (ix <= size_C1_2 && iy <= size_C2_2)
        C2[ix+1, iy+1] = C[ix+1, iy+1] - dt * ((@qx(ix + 1, iy) - @qx(ix, iy)) * _dx + (@qy(ix, iy + 1) - @qy(ix, iy)) * _dy)
    end
    return
end

@views function diffusion_2D(nx, ny, hide_comm_1 = 8, hide_comm_2 = 2; do_visu=false, do_save = false, strong_scaling = false, weak_scaling = false, hide_communication = false, init_hide_comm = false)
    # Physics
    Lx, Ly = 10.0, 10.0
    D      = 1.0
    ttot = 1e0
    nout   = 20
    # Derived numerics
    me, dims = init_global_grid(nx, ny, 1; init_MPI=false)
    dx, dy   = Lx / nx_g(), Ly / ny_g()
    dt     = min(dx, dy)^2 / D / 4.1
    if weak_scaling || hide_communication
        nt     = 200
    else
        nt     = 500
    end
    xc, yc = LinRange(dx / 2, Lx - dx / 2, nx), LinRange(dy / 2, Ly - dy / 2, ny)
    D_dx   = D / dx
    D_dy   = D / dy
    _dx, _dy = 1.0 / dx, 1.0 / dy
    # Array initialisation
    C       = @zeros(nx, ny)
    C      .= Data.Array([exp(-(x_g(ix,dx,C)+dx/2 -Lx/2)^2 -(y_g(iy,dy,C)+dy/2 -Ly/2)^2) for ix=1:size(C,1), iy=1:size(C,2)])
    C2      = copy(C)
    size_C1_2, size_C2_2 = size(C, 1) - 2, size(C, 2) - 2
    t_tic  = 0.0
    niter  = 0
    if do_visu
        if (me==0) ENV["GKSwstype"]="nul"; if isdir("viz2D_mxpu_out")==false mkdir("viz2D_mxpu_out") end; loadpath = "./viz2D_mxpu_out/"; anim = Animation(loadpath,String[]); println("Animation directory: $(anim.dir)") end
        nx_v, ny_v = (nx-2)*dims[1], (ny-2)*dims[2]
        if (nx_v*ny_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        C_v   = zeros(nx_v, ny_v) # global array for visu
        C_inn = zeros(nx-2, ny-2) # no halo local array for visu
        xi_g, yi_g = LinRange(dx+dx/2, Lx-dx-dx/2, nx_v), LinRange(dy+dy/2, Ly-dy-dy/2, ny_v) # inner points only
    end
    if do_save
        C_save_inn = zeros(nx-2, ny-2)
        C_v_save   = zeros(nx_v, ny_v)
    end
    # Time loop
    for it = 1:nt
        if (it == 11) t_tic = Base.time(); niter = 0 end
        if init_hide_comm
            @parallel compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size_C1_2, size_C2_2)
            C, C2 = C2, C # pointer swap
            update_halo!(C)
        else
            @hide_communication (hide_comm_1, hide_comm_2) begin
                @parallel compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size_C1_2, size_C2_2)
                C, C2 = C2, C # pointer swap
                update_halo!(C)
            end
        end
        niter += 1
        # Visualize
        if do_save 
            C_save_inn .= Array(C)[2:end-1,2:end-1]; gather!(C_inn, C_v_save)
        end
        if do_visu  && (it % nout == 0)
            C_inn .= Array(C)[2:end-1,2:end-1]; gather!(C_inn, C_v)
            if (me==0)
                opts = (aspect_ratio=1, xlims=(xi_g[1], xi_g[end]), ylims=(yi_g[1], yi_g[end]), clims=(0.0, 1.0), c=:turbo, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
                heatmap(xi_g, yi_g, Array(C_v)'; opts...); frame(anim)
            end
        end
    end 
    if (do_visu && me==0) gif(anim, "diffusion_2D_mxpu.gif", fps = 5)  end
    t_toc = Base.time() - t_tic
    A_eff = 2 / 1e9 * nx * ny * sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc / niter                  # Execution time per iteration [s]
    T_eff = A_eff / t_it                   # Effective memory throughput [GB/s]
    @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter)
    if (do_save && me==0) file = matopen("./2D-diffusion-implicit-global/implicitglob2D_out_C_$(file_name).mat", "w"); write(file, "C", Array(C_v_save)); close(file) end
    finalize_global_grid(;finalize_MPI = false)
    if hide_communication
        return t_toc 
    else
        return T_eff
    end
end

strong_scaling     = false
weak_scaling       = false
hide_communication = true

if strong_scaling
    nx = ny = 16 * 2 .^ (1:10)
    T_eff = zeros(size(nx, 1))
    for i = 1:size(nx, 1)
        T_eff[i] = diffusion_2D(nx[i], ny[i]; do_visu=false, strong_scaling = strong_scaling)
    end
    plot(nx, T_eff, title = "Strong Scaling - Memory Throughput", xlabel = "nx", ylabel = "Memory Throughput (GB/s)", linewidth = 2, xscale=:log10)
    savefig("./docs/strong_scaling_2.png")
elseif weak_scaling 
    nx = ny = 16384
    diffusion_2D(nx, ny; do_visu=false, weak_scaling = weak_scaling)
elseif hide_communication
    nx = ny         = 16384
    init_hide_comm_ = true
    t_vector        = zeros(5)
    norm_time       = 1.756
    t_vector[1]     = diffusion_2D(nx, ny; do_visu=false, hide_communication = hide_communication, init_hide_comm = init_hide_comm_) / norm_time
    # t_vector[3] = 5.944 / norm_time;
    init_hide_comm_ = false
    hide_comm_1     = [2, 8, 16, 16]
    hide_comm_2     = [2, 2, 4, 16]
    # t_vector[2] = diffusion_2D(nx, ny, hide_comm_1[1], hide_comm_2[1]; do_visu=false, hide_communication = hide_communication, init_hide_comm = init_hide_comm) / norm_time
    for i in 2:5
        t_vector[i] = diffusion_2D(nx, ny, hide_comm_1[i-1], hide_comm_2[i-1]; do_visu=false, hide_communication = hide_communication, init_hide_comm = init_hide_comm_) / norm_time
        # t_vector[i] = t_vector[i] / norm_time
    end
    tex_hide_comm = ["no-hidecomm", "(2,2)", "(8,2)", "(16,4)", "(16,16)"]
    plot(tex_hide_comm, t_vector, title = "Effect of Hide Communication", xlabel = "Hide communication parameters", ylabel = "Normalized Execution Time", linewidth = 2, ylims=(0.5,1.25))
    savefig("./docs/hide_communication_2.png")
else
    nx = ny = 8192
    diffusion_2D(nx, ny; do_visu=false, do_save = false)
end

MPI.Finalize()