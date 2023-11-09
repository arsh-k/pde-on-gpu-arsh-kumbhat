using CUDA, BenchmarkTools

# Defining triad kernel function.
@inbounds function memcopy_triad_KP!(A, B, C, s)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    A[ix,iy] = B[ix,iy] + s*C[ix,iy]
    return nothing
end

max_threads  = attribute(device(),CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
array_sizes = []
throughputs = []
# Evaluating the nx and ny for maximum memory throughputs.
for pow = 0:11
    nx = ny = 32*2^pow
    if (3*nx*ny*sizeof(Float64) > CUDA.available_memory()) break; end
    A = CUDA.zeros(Float64, nx, ny);
    B = CUDA.rand(Float64, nx, ny);
    t_it = @belapsed begin copyto!($A, $B); synchronize() end
    T_tot = 2*1/1e9*nx*ny*sizeof(Float64)/t_it
    push!(array_sizes, nx)
    push!(throughputs, T_tot)
    println("(nx=ny=$nx) T_tot = $(T_tot)")
    CUDA.unsafe_free!(A)
    CUDA.unsafe_free!(B) #Releases array memory to reduce pressue on memory allocator.
end

s = rand()
T_tot_max, index = findmax(throughputs)
nx = ny = array_sizes[index]
thread_count = []
throughputs  = []
println("-----CHECKING THREADS-----")
# Evaluating thread and block combination for T_peak.
for pow = 0:Int(log2(max_threads/32))
    threads = (32, 2^pow)
    A = CUDA.zeros(Float64, nx, ny);
    B = CUDA.rand(Float64, nx, ny);
    C = CUDA.rand(Float64, nx, ny);
    blocks  = (nx÷threads[1], ny÷threads[2])
    t_it = @belapsed begin @cuda blocks=$blocks threads=$threads memcopy_triad_KP!($A, $B, $C, $s); synchronize() end
    T_tot = 3*1/1e9*nx*ny*sizeof(Float64)/t_it
    push!(thread_count, prod(threads))
    push!(throughputs, T_tot)
    println("(threads=$threads) T_tot = $(T_tot)")
end

A = CUDA.zeros(Float64, nx, ny);
B = CUDA.rand(Float64, nx, ny);
C = CUDA.rand(Float64, nx, ny);
T_tot_max, index = findmax(throughputs)
threads = (32, thread_count[index]÷32)
blocks  = (nx÷threads[1], ny÷threads[2])
t_it = @belapsed begin @cuda blocks=$blocks threads=$threads memcopy_triad_KP!($A, $B, $C, $s); synchronize() end
T_tot = 3*1/1e9*nx*ny*sizeof(Float64)/t_it
println(T_tot)