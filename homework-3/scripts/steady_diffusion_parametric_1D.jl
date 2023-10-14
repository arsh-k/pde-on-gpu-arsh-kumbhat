using Plots, Plots.Measures, Printf
default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function steady_diffusion_1D()
    # physics
    lx      = 20.0
    dc      = 1.0
    # numerics
    nx      = 100
    ϵtol    = 1e-8
    maxiter = 100nx
    ncheck  = ceil(Int, 0.25nx)
    fact = 0.5:0.1:1.5
    conv = zeros(size(fact))
    # derived numerics
    dx      = lx / nx
    xc      = LinRange(dx / 2, lx - dx / 2, nx)
    # parameter search loop
    for ifact in eachindex(fact)
        re      = 2π*fact[ifact]
        ρ       = (lx/(dc*re))^2
        dτ      = dx / sqrt(1 / ρ)
        # array initialisation
        C       = @. 1.0 + exp(-(xc - lx / 4)^2) - xc / lx
        qx      = zeros(Float64, nx - 1)
        # iteration loop
        iter = 1; err = 2ϵtol; iter_evo = Float64[]; err_evo = Float64[]
        while err >= ϵtol && iter <= maxiter
            qx         .-= dτ ./ (ρ * dc .+ dτ) .* (qx .+ dc .* diff(C) ./ dx)
            C[2:end-1] .-= dτ .* diff(qx) ./ dx
            if iter % ncheck == 0
                err = maximum(abs.(diff(dc .* diff(C) ./ dx) ./ dx))
                push!(iter_evo, iter / nx); push!(err_evo, err)
            end
            iter += 1
        end
        conv[ifact] = iter/nx
    end
    p1 = plot(fact,conv; grid = true, marker=(:circle,10),
    xlabel="fact", ylabel="iter/nx", title="Optimal Parameter Estimation")
    display(p1)
    savefig("parametric_estimation.png")
end

steady_diffusion_1D()
