# Visualisation script for the 2D MPI solver
using Plots, MAT

nprocs = (2, 2) # nprocs (x, y) dim

@views function vizme2D_mpi(nprocs, it)
    C = []; ip = 1
    for ipx = 1:nprocs[1]
        for ipy = 1:nprocs[2]
            file = matopen("../data/2D-diffusion-mpi-gpu/mpi2D_out_C_$(ip-1)_$(it).mat")
            C_loc = read(file, "C"); close(file)
            nx_i, ny_i = size(C_loc, 1) - 2, size(C_loc, 2) - 2
            ix1, iy1 = 1 + (ipx - 1) * nx_i, 1 + (ipy - 1) * ny_i
            if (ip == 1) C = zeros(nprocs[1] * nx_i, nprocs[2] * ny_i) end
            C[ix1:ix1+nx_i-1, iy1:iy1+ny_i-1] .= C_loc[2:end-1, 2:end-1]
            ip += 1
        end
    end
    fontsize = 12
    opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
            ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier",
            xlabel="Lx", ylabel="Ly", xlims=(1, size(C, 1)), ylims=(1, size(C, 2)), clims=(0,1))
    display(heatmap(C'; c=:turbo, title="diffusion 2D MPI - GPU", opts...))
    return
end

nvis = 5
nt   = 100
anim = @animate for it = 1:nt
    vizme2D_mpi(nprocs, it)
end every nvis
gif(anim, "../docs/diffusion_2D_mpi_gpu.gif")