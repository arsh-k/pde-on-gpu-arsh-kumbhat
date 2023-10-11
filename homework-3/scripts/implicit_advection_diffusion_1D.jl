using Plots, Plots.Measures, Printf
default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function implicit_adv_diff_1D()
    # physics
    lx      = 20.0
    dc      = 1.0
    vx      = 1.0
    # numerics
    nx      = 100
    ϵtol    = 1e-8
    maxiter = 50nx
    ncheck  = ceil(Int, 0.25nx)
    nt      = 10
    # derived numerics and physics
    dx      = lx / nx
    dt      = dx/abs(vx)
    da      = lx^2/dc/dt
    re      = π + sqrt(π^2 + da)
    ρ       = (lx / (dc * re))^2
    xc      = LinRange(dx / 2, lx - dx / 2, nx)
    dτ      = dx / sqrt(1 / ρ)
    # array initialisation
    C       = @. 1.0 + exp(-(xc - lx / 4)^2) - xc / lx
    C_old   = copy(C)
    C_i     = copy(C)
    qx      = zeros(Float64, nx - 1)
    # physical time loop 
    for it = 1:nt
        C_old .= C
        # iteration loop
        iter = 1; err = 2ϵtol; iter_evo = Float64[]; err_evo = Float64[]
        while err >= ϵtol && iter <= maxiter
            qx         .-= dτ ./ (ρ .+ dτ / dc) .* (qx ./ dc .+ diff(C) ./ dx)
            C[2:end-1] .-= dτ ./ (1.0 .+ dτ / dt) .* ((C[2:end-1] - C_old[2:end-1]) ./ dt .+ diff(qx) ./ dx)
            if iter % ncheck == 0
                err = maximum(abs.(diff(dc .* diff(C) ./ dx) ./ dx .- (C[2:end-1] - C_old[2:end-1]) ./ dt))
                push!(iter_evo, iter / nx); push!(err_evo, err)
            end
            iter += 1
        end
        C[2:end]    -= max(vx, 0.0) .* dt .* diff(C) ./dx
        C[1:end-1]  -= min(vx, 0.0) .* dt .* diff(C) ./dx
    end
    display(plot(xc, [C_i, C]; xlims=(0, lx), ylims=(-0.1, 2.0), label = ["Initial Concentration" "Final Concentration"],
    xlabel="lx", ylabel="Concentration", title = "Implicit 1-D Advection-Diffusion"))
    savefig("implicit_advection_diffusion_1D.png")
end

implicit_adv_diff_1D()
