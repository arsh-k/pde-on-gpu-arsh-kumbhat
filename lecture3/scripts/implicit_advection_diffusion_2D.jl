using Plots, Plots.Measures, Printf
default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function steady_diffusion_1D()
    # physics
    lx,ly   = 10.0,10.0
    dc      = 1.0
    vx      = 10.0
    vy      = -10.0
    # numerics
    nx,ny   = 200,201
    ϵtol    = 1e-8
    maxiter = 10nx
    ncheck  = ceil(Int,0.02nx)
    nt      = 50
    # derived numerics and physics
    dx      = lx / nx
    dy      = ly / ny
    dt = min(dx/abs(vx),dy/abs(vy))/2
    da      = lx^2/dc/dt
    re      = π + sqrt(π^2 + da)
    ρ       = (lx / (dc * re))^2
    xc      = LinRange(dx / 2, lx - dx / 2, nx)
    yc      = LinRange(dy / 2, ly - dy / 2, ny)
    dτ      = min(dx,dy)/sqrt(1/ρ)/sqrt(2)
    # array initialisation
    C       = @. exp(-(xc-lx/4)^2 -(yc'-3ly/4)^2) #(200,201)
    C_old   = copy(C)
    qx      = zeros(Float64, nx - 1, ny) #(199,201)
    qy      = zeros(Float64, nx, ny - 1) #(200,200)
    # iteration loop
    anim = @animate for it = 1:nt
        C_old .= C
        iter = 1; err = 2ϵtol; iter_evo = Float64[]; err_evo = Float64[]
        while err >= ϵtol && iter <= maxiter
            qx              .-= dτ ./ (ρ .+ dτ / dc) .* (qx ./ dc .+ diff(C, dims = 1) ./ dx)
            qy              .-= dτ ./ (ρ .+ dτ / dc) .* (qy ./ dc .+ diff(C, dims = 2) ./ dy)
            ∇q                = diff(qx, dims = 1)[:,2:end-1] ./ dx 
            ∇q               += diff(qy, dims = 2)[2:end-1,:] ./ dy
            C[2:end-1,2:end-1]   .-= dτ ./ (1.0 .+ dτ / dt) .* ((C[2:end-1,2:end-1] - C_old[2:end-1,2:end-1]) ./ dt .+ ∇q)
            if iter % ncheck == 0
                err = maximum(abs.(diff(dc .* diff(C, dims = 1) ./ dx, dims = 1)[:,2:end-1] ./ dx + diff(dc .* diff(C, dims = 2) ./ dy, dims = 2)[2:end-1,:] ./ dy .- (C[2:end-1,2:end-1] - C_old[2:end-1,2:end-1]) ./ dt))
                push!(iter_evo, iter / nx); push!(err_evo, err)
                # visualisation
                p1 = heatmap(xc,yc,C';xlims=(0,lx),ylims=(0,ly),clims=(0,1),aspect_ratio=1,
                xlabel="lx",ylabel="ly",title="iter/nx=$(round(iter/nx,sigdigits=3))")
                p2 = plot(iter_evo,err_evo;xlabel="iter/nx",ylabel="err",
                yscale=:log10,grid=true,markershape=:circle,markersize=10)
                display(plot(p1,p2;layout=(2,1)))
            end
            iter += 1
        end
        #(199,200)
        # ∇C = diff(C, dims = 1)[:2:end-1] ./dx
        C[2:end, :]     -= max(vx, 0.0) .* dt .* diff(C, dims = 1) ./dx
        C[1:end-1, :]   -= min(vx, 0.0) .* dt .* diff(C, dims = 1) ./dx
        C[:, 2:end]     -= max(vy, 0.0) .* dt .* diff(C, dims = 2) ./dy
        C[:, 1:end-1]   -= min(vy, 0.0) .* dt .* diff(C, dims = 2) ./dy
    end
    gif(anim, "implicit_advection_diffusion_2D.gif"; fps = 2)
end

steady_diffusion_1D()
