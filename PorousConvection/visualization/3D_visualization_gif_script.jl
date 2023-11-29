using GLMakie, Printf 

function load_array(Aname, A)
    fname = string(Aname, ".bin")
    fid=open(fname, "r"); read!(fid, A); close(fid)
end

function visualise(iframe = 0)
    lx, ly, lz = 40.0, 20.0, 20.0
    nx = 506
    ny = nz = 250
    T  = zeros(Float32, nx, ny, nz)
    load_array(@sprintf("out_T_%04d", iframe), T)
    xc, yc, zc = LinRange(0, lx, nx), LinRange(0, ly, ny), LinRange(0, lz, nz)
    fig = Figure(resolution=(1600, 1000), fontsize=24)
    ax  = Axis3(fig[1, 1]; aspect=(1, 1, 0.5), title="Temperature", xlabel="lx", ylabel="ly", zlabel="lz")
    surf_T = contour!(ax, xc, yc, zc, T; alpha=0.05, colormap=:turbo)
    save(@sprintf("../3D_gif/%06d.png", iframe), fig)
    return fig
end

nvis    = 1
nt      = 20
for it = 1:nt
    visualise(it)
end 
# gif(anim, "../3D_gif/porous_convection_3D_multixpu.gif")
# fig = Figure(resolution=(1600, 1000), fontsize=24)
# record(visualise, fig, "porous_convection_3D_multixpu.mp4", framerate = 1)

import Plots:Animation, buildanimation  
nframes = 20                
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("../3D_gif/", fnames); 
buildanimation(anim, "../3D_gif/porous_convection_3D_multixpu.gif", fps = 10, show_msg=false) 