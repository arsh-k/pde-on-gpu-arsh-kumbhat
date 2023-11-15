using Plots, Plots.Measures
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

weak_scaling_times = [0.051, 0.317, 0.1462, 2.29, 5.944]
num_procs = [1, 4, 16, 25, 64]
speedup = zeros(size(weak_scaling_times))
speedup .= weak_scaling_times ./ weak_scaling_times[1]

plot(num_procs, speedup, title = "Weak Scaling: Speed-up", xlabel = "nprocs", ylabel = "Speed-up", 
    linewidth = 2)
savefig("../docs/weak_scaling.png")