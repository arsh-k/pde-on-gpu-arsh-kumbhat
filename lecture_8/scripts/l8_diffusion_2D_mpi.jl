# # 2D linear diffusion Julia MPI solver
# run: ~/.julia/bin/mpiexecjl -n 4 julia --project l8_diffusion_2D_mpi.jl
using Plots, Printf, MAT
import MPI

# enable plotting by default
if !@isdefined do_save; do_save = true end

# MPI functions
@views function update_halo!(A, neighbors_x, neighbors_y, comm)
    # Send to / receive from neighbor 1 in dimension x ("left neighbor")
    if neighbors_x[1] != MPI.PROC_NULL
        sendbuf = A[2, :]
        recvbuf = zeros(size(A, 2))
        # For GPU
        # sendbuf = zeros(size(A,2))
        # copyto!(sendbuf, A[2,:])
        # recvbuf = zeros(size(A, 2))
        MPI.Send(sendbuf,  neighbors_x[1], 0, comm) #blocking function (helps hiding latency)
        MPI.Recv!(recvbuf, neighbors_x[1], 1, comm)
        A[1, :] .= recvbuf
        #copy!(A[1,:], recvbuf)    
    end
    # Send to / receive from neighbor 2 in dimension x ("right neighbor")
    if neighbors_x[2] != MPI.PROC_NULL
        sendbuf = A[end - 1, :]
        recvbuf = zeros(size(A, 2))
        MPI.Recv!(recvbuf, neighbors_x[2], 0, comm)
        MPI.Send(sendbuf,  neighbors_x[2], 1, comm)
        A[end, :] .= recvbuf    
    end
    # Send to / receive from neighbor 1 in dimension y ("bottom neighbor")
    if neighbors_y[1] != MPI.PROC_NULL
        sendbuf = A[:, 2]
        recvbuf = zeros(size(A, 1))
        MPI.Send(sendbuf,  neighbors_y[1], 0, comm)
        MPI.Recv!(recvbuf, neighbors_y[1], 1, comm)
        A[:, 1] .= recvbuf    
    end
    # Send to / receive from neighbor 2 in dimension y ("top neighbor")
    if neighbors_y[2] != MPI.PROC_NULL
        sendbuf = A[:, end - 1]
        recvbuf = zeros(size(A, 1))
        MPI.Recv!(recvbuf, neighbors_y[2], 0, comm)
        MPI.Send(sendbuf,  neighbors_y[2], 1, comm)
        A[:, end] .= recvbuf    
    end
    return
end

@views function diffusion_2D_mpi(; do_save=false)
    # MPI
    MPI.Init()
    dims   = [0, 0]
    comm   = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)
    MPI.Dims_create!(nprocs, dims)
    comm_cart = MPI.Cart_create(comm, dims, [0, 0], 1)
    me     = MPI.Comm_rank(comm_cart)
    coords = MPI.Cart_coords(comm_cart)
    neighbors_x = MPI.Cart_shift(comm_cart, 0, 1) #The first int argument is the direction of the shift
    neighbors_y = MPI.Cart_shift(comm_cart, 1, 1) #The second int argument is the amount of shift, 1 indicates the immediate neighbours.
    (me == 0) && println("nprocs=$(nprocs), dims[1]=$(dims[1]), dims[2]=$(dims[2])")
    # Physics
    lx, ly = 10.0, 10.0
    D      = 1.0
    nt     = 100
    # Numerics
    nx, ny = 32, 32                             # local number of grid points
    nx_g, ny_g = dims[1] * (nx - 2) + 2, dims[2] * (ny - 2) + 2 # global number of grid points
    nvis   = 5
    nprocs_vis = (2, 2) # nprocs (x, y) dim
    # Derived numerics
    dx, dy = lx / nx_g, ly / ny_g                   # global
    dt     = min(dx, dy)^2 / D / 4.1
    # Array allocation
    qx     = zeros(nx - 1, ny - 2)
    qy     = zeros(nx - 2, ny - 1)
    # Initial condition
    x0, y0 = coords[1] * (nx - 2) * dx, coords[2] * (ny - 2) * dy
    xc     = [x0 + ix * dx - dx / 2 - 0.5 * lx for ix = 1:nx]
    yc     = [y0 + iy * dy - dy / 2 - 0.5 * ly for iy = 1:ny]
    C      = exp.(.-xc .^ 2 .- yc' .^ 2)
    t_tic = 0.0
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time() end
        qx .= .-D * diff(C[:, 2:end-1], dims=1) / dx
        qy .= .-D * diff(C[2:end-1, :], dims=2) / dy
        C[2:end-1, 2:end-1] .= C[2:end-1, 2:end-1] .- dt * (diff(qx, dims=1) / dx .+ diff(qy, dims=2) / dy)
        update_halo!(C, neighbors_x, neighbors_y, comm_cart)
        if do_save file = matopen("../data/2D-diffusion-gif/mpi2D_out_C_$(me)_$(it).mat", "w"); write(file, "C", Array(C)); close(file) end
    end 
    t_toc = (Base.time() - t_tic)
    (me == 0) && @printf("Time = %1.4e s, T_eff = %1.2f GB/s \n", t_toc, round((2 / 1e9 * nx * ny * sizeof(lx)) / (t_toc / (nt - 10)), sigdigits=2))
    # Save to visualise
    MPI.Finalize()
    return
end

diffusion_2D_mpi(; do_save=do_save)