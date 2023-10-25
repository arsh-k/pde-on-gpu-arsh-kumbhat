using Plots, Plots.Measures, Printf, BenchmarkTools, LoopVectorization
default(size=(600, 500), framestyle=:box, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

function memcopy(;bench) #After semicolon, we add kwargs
    # # Numerics
    # nx, ny  = 512, 512
    # nt      = 2e4
    # array programming function
    function compute_ap!(C2, C, A)
        C2 .= C .+ A
        return nothing
    end
    # kernel programming function
    function compute_kp!(C2, C, A)
        nx, ny = size(C)
        @tturbo for ix = 1:nx
            for iy = 1:ny
                @inbounds C2[ix, iy] = C[ix, iy] + A[ix, iy]
            end
        end
    end
    pow_vect      = 1:8
    T_eff_vect_ap = zeros(Float64, size(pow_vect))
    T_eff_vect_kp = zeros(Float64, size(pow_vect))
    domain_size   = zeros(Float64, size(pow_vect))  
    for pow in pow_vect
        # numerics
        nx = ny          = 16 * 2 ^ pow
        nt               = max(nx, ny)
        domain_size[pow] = nx*ny
        # array initialisation
        C                = rand(Float64, nx, ny)
        C2               = copy(C)
        A                = copy(C)
        # Array Programming Memory Throughput evaluation
        if bench == :loop
            # iteration loop
            t_tic = 0.0; niter = 0
            for iter=1:nt
                if iter == 11 #Warming up the function before you track the time.
                    t_tic = Base.time(); niter = 0
                end
                compute_ap!(C2, C, A)
                niter += 1 
            end
            t_toc = Base.time() - t_tic
            t_it  = t_toc/niter
            println("------LOOP($(pow))------")
        elseif bench == :btool
            t_toc = @belapsed $compute_ap!($C2, $C, $A)
            t_it  = t_toc
            println("------BENCHMARKTOOLS($(pow))------")
        end
        size_f = sizeof(Float64)
        A_eff = 3*size_f*nx*ny                                                                 # Effective main memory access per iteration [GB] 
        T_eff = A_eff/t_it                                                                     # Effective memory throughput [GB/s]
        @printf("Array Programming: Time per iteration = %1.5f sec(@ %1.2f GB/s) \n", t_it, T_eff / 1e9)
        T_eff_vect_ap[pow] = T_eff / 1e9
        # Kernel Programming Memory Throughput evaluation
        if bench == :loop
            # iteration loop
            t_tic = Base.time(); niter = 0
            for iter=1:nt
                if iter == 11 #Warming up the function before you track the time.
                    t_tic = Base.time(); niter = 0
                end
                compute_kp!(C2, C, A)
                niter += 1 
            end
            t_toc = Base.time() - t_tic
            t_it  = t_toc/niter
        elseif bench == :btool
            t_toc = @belapsed $compute_ap!($C2, $C, $A)
            t_it  = t_toc
        end
        size_f = sizeof(Float64)
        A_eff  = 3*size_f*nx*ny                                                                 # Effective main memory access per iteration [GB] 
        T_eff  = A_eff/t_it                                                                     # Effective memory throughput [GB/s]
        @printf("Kernel Programming: Time per iteration = %1.5f sec(@ %1.2f GB/s) \n", t_it, T_eff / 1e9)
        T_eff_vect_kp[pow] = T_eff / 1e9
    end
    return T_eff_vect_ap, T_eff_vect_kp
end

# Plotting comparison of memory throughputs between array and kernel programming.
T_eff_vect_ap_btool, T_eff_vect_kp_btool = memcopy(bench = :btool)
T_eff_vect_ap_loop, T_eff_vect_kp_loop   = memcopy(bench = :loop)
nx_ = ny_   = 16 * 2 .^ (1:8)
domain_size = nx_ .* ny_
max_pos_ap_loop = argmax(T_eff_vect_ap_loop); max_pos_kp_loop = argmax(T_eff_vect_kp_loop)
max_ap_loop = maximum(T_eff_vect_ap_loop); max_kp_loop = maximum(T_eff_vect_kp_loop)
max_bandwidth = 37.5 
display(plot(domain_size, [T_eff_vect_ap_btool,T_eff_vect_kp_btool, T_eff_vect_ap_loop, T_eff_vect_kp_loop], 
        xlabel = "Domain Size", ylabel = "Memory Throughput (GB/s)", 
        label = ["BTool: Array Programming" "BTool: Kernel Programming" "Loop: Array Programming" "Loop: Kernel Programming"],
        linewidth = 2, xscale=:log10, title = "Memory Throughput Evaluation (memcopy)"))
display(scatter!((domain_size[max_pos_kp_loop], max_kp_loop), label = "Loop: Kernel Programming Maximum"))
display(scatter!((domain_size[max_pos_ap_loop], max_ap_loop), label = "Loop: Array Programming Maximum"))
hline!([max_bandwidth], label= "Maximum CPU Bandwidth",line=(:dot, 1), linewidth = 2, linecolor = "green")
savefig("../docs/memcopy_ex_2_task_2.png")