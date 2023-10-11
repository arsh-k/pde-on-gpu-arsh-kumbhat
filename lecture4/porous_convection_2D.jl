using Plots, Plots.Measures, Printf
default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function porous_convection_2_dim()
    # physics
    lx      = 40.0
    ly      = 20.0
    k_ηf    = 1.0
    cfl = 1.0/sqrt(2.1) # 1D - 1.1, 2D - 2.1, 3D - 3.1
    # numerics
    nx      = 100
    ny      = ceil(Int, nx*ly/lx)
    ϵtol    = 1e-8
    maxiter = 100max(nx,ny)
    ncheck  = ceil(Int, 0.25max(nx,ny))
    re      = 2π
    # derived numerics
    dx      = lx / nx
    dy      = ly / ny
    xc      = LinRange(dx / 2, lx - dx / 2, nx)
    yc      = LinRange(dy / 2, ly - dy / 2, ny)
    θ_dτ    = max(lx, ly)/re/cfl/min(dx, dy)
    β_dτ    = (re*k_ηf)/(cfl*min(dx, dy)*max(lx, ly))    
    # array initialisation
    Pf       = @. exp(-(xc-lx/4)^2 -(yc'-ly/4)^2)
    qDx      = zeros(Float64, nx + 1, ny)
    qDy      = zeros(Float64, nx, ny+1) #Endpoints are zero fluxes.
    r_Pf     = zeros(Float64, nx, ny)
    # iteration loop
    iter = 1; err_Pf = 2ϵtol; 
    while err_Pf >= ϵtol && iter <= maxiter
        qDx[2:end-1, :] .-=  (qDx[2:end-1, :] .+ k_ηf .* diff(Pf, dims = 1) ./ dx)./ (1.0 .+ θ_dτ)
        qDy[:, 2:end-1] .-=  (qDy[:, 2:end-1] .+ k_ηf .* diff(Pf, dims = 2) ./ dy)./ (1.0 .+ θ_dτ)
        Pf              .-=  (diff(qDx, dims = 1) ./ dx) ./β_dτ + (diff(qDy, dims = 2) ./ dy) ./ β_dτ
        if iter % ncheck == 0
            r_Pf  .= diff(qDx, dims = 1)./dx + diff(qDy, dims = 2) ./dy
            err_Pf = maximum(abs.(r_Pf))
            @printf("  iter/nx=%.1f, err_Pf=%1.3e\n",iter/nx,err_Pf)      
        end
        p1 = heatmap(xc,yc,Pf';xlims=(0,lx),ylims=(0,ly),clims=(0,1),aspect_ratio=1,
        xlabel="lx",ylabel="ly",title="iter/nx=$(round(iter/nx,sigdigits=3))")
        display(plot(p1))
        iter += 1
    end
end

porous_convection_2_dim()
