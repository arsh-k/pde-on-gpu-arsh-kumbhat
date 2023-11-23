using Plots, Plots.Measures
default(size=(600, 500), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=11, tickfontsize=11, titlefontsize=11)

weak_scaling_times = [1.756, 1.83, 1.873, 1.896, 1.939]
num_procs          = [1, 4, 16, 25, 64]
speedup            = zeros(size(weak_scaling_times))
speedup           .= weak_scaling_times ./ weak_scaling_times[1]

plot(num_procs, speedup, title = "Weak Scaling", xlabel = "nprocs", ylabel = "Normalized Execution Time", 
    linewidth = 2, ylims =(0,1.25))
savefig("./docs/weak_scaling.png")