using Plots, Plots.Measures, Printf
default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function porous_convection_2_dim()
    # physics
    lx        = 40.0
    ly        = 20.0
    k_ηf      = 1.0
    cfl       = 1.0/sqrt(2.1) # 1D - 1.1, 2D - 2.1, 3D - 3.1
    αρgx,αρgy = 0.0,1.0
    αρg       = sqrt(αρgx^2+αρgy^2)
    ΔT        = 200.0
    ϕ         = 0.1
    Ra        = 1000
    λ_ρCp     = 1/Ra*(αρg*k_ηf*ΔT*ly/ϕ) # Ra = αρg*k_ηf*ΔT*ly/λ_ρCp/ϕ
    # numerics
    nx        = 127
    ny        = ceil(Int, nx*ly/lx)
    ϵtol      = 1e-8
    maxiter   = 100max(nx,ny)
    ncheck    = ceil(Int, 0.25max(nx,ny))
    re        = 2π
    nt        = 500
    nvis      = 5
    # derived numerics
    dx        = lx / nx
    dy        = ly / ny
    xc        = LinRange(-lx/2 + dx / 2, lx/2 - dx / 2, nx)
    yc        = LinRange(-ly + dy / 2, -dy/2, ny)
    θ_dτ      = max(lx, ly)/re/cfl/min(dx, dy)
    β_dτ      = (re*k_ηf)/(cfl*min(dx, dy)*max(lx, ly))    
    dt_diff   = min(dx,dy)^2/λ_ρCp/4.1
    # array initialisation
    Pf        = @. exp(-(xc-lx/4)^2 -(yc'-ly/4)^2)
    qDx       = zeros(Float64, nx + 1, ny)
    qDy       = zeros(Float64, nx, ny+1) #Endpoints are zero fluxes.
    r_Pf      = zeros(Float64, nx, ny)
    T         = @. ΔT*exp(-xc^2 - (yc'+ly/2)^2)
    T[:,1] .= ΔT/2; T[:,end] .= -ΔT/2
    T[[1,end],:] .= T[[2,end-1],:]
    dT_dtadv  = zeros(Float64, nx-2, ny-2)
    qTx       = zeros(Float64, nx+1, ny) #Fluxes are zero at the boundaries in the x-direction.
    qTy       = zeros(Float64, nx, ny-1) #Constant temperatures at the y-direction boundaries.
    # visualisation init
    st        = ceil(Int,nx/25)
    Xc, Yc    = [x for x=xc, y=yc], [y for x=xc,y=yc]
    Xp, Yp    = Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end]
    qDxc      = zeros(Float64, nx, ny)
    qDyc      = zeros(Float64, nx, ny)
    qDmag     = zeros(Float64, nx, ny)
    # time loop
    anim = @animate for it = 1:nt
        iter = 1; err_Pf = 2ϵtol;
        # iteration loop 
        while err_Pf >= ϵtol && iter <= maxiter
            qDx[2:end-1, :]  .-=  (qDx[2:end-1, :] .+ k_ηf .* diff(Pf, dims = 1) ./ dx)./ (1.0 .+ θ_dτ)
            qDy[:, 2:end-1]  .-=  (qDy[:, 2:end-1] .+ k_ηf .* diff(Pf, dims = 2) ./ dy)./ (1.0 .+ θ_dτ)
            #Boussinesq approximation
            qDx[2:end-1,:]    .= qDx[2:end-1,:] .+ (k_ηf * αρgx .* (T[1:end-1,:] + T[2:end,:] ./2) ./ (1.0 .+ θ_dτ))
            qDy[:,2:end-1]    .= qDy[:,2:end-1] .+ (k_ηf * αρgy .* (T[:,1:end-1] + T[:,2:end] ./2)  ./ (1.0 .+ θ_dτ))
            Pf               .-=  (diff(qDx, dims = 1) ./ dx) ./β_dτ + (diff(qDy, dims = 2) ./ dy) ./ β_dτ
            if iter % ncheck == 0
                r_Pf  .= diff(qDx, dims = 1)./dx .+ diff(qDy, dims = 2) ./dy
                err_Pf = maximum(abs.(r_Pf))
                @printf("it = %d, iter/nx=%.1f, err_Pf=%1.3e\n",it,iter/nx,err_Pf)
            end
            iter += 1
        end
        dt_adv                    = ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1
        dt                        = min(dt_diff,dt_adv)
        qTx[2:end-1,:]           .= -λ_ρCp.*diff(T, dims = 1)
        qTy                      .= -λ_ρCp.*diff(T, dims = 2)
        dT_dtadv                 .= 0.0
        T[:,2:end-1]            .-= dt.*(diff(qTx[:,2:end-1], dims =1)./dx + diff(qTy, dims =2)./dy) 
        dT_dtadv                .-= dt ./ ϕ .*max.(qDx[2:end-2, 2:end-1],0.0) .* diff(T[1:end-1, 2:end-1], dims = 1)./dx
        dT_dtadv                .-= dt ./ ϕ .*min.(qDx[3:end-1, 2:end-1],0.0) .* diff(T[2:end, 2:end-1], dims = 1)./dx
        dT_dtadv                .-= dt ./ ϕ .*max.(qDy[2:end-1, 2:end-2],0.0) .* diff(T[2:end-1, 1:end-1], dims = 2)./dy
        dT_dtadv                .-= dt ./ ϕ .*min.(qDy[2:end-1, 3:end-1],0.0) .* diff(T[2:end-1, 2:end], dims = 2)./dy
        T[2:end-1,2:end-1]      .+= dT_dtadv
        # visualisation
        qDxc   .= 0.5 .* (qDx[1:end-1,:] .+ qDx[2:end,:])
        qDyc   .= 0.5 .* (qDy[:,1:end-1] .+ qDy[:,2:end])
        qDmag  .= sqrt.(qDxc.^2 .+ qDyc.^2)
        qDxc  ./= qDmag
        qDyc  ./= qDmag
        qDx_p   = qDxc[1:st:end,1:st:end]
        qDy_p   = qDyc[1:st:end,1:st:end]
        p1      = heatmap(xc,yc,T';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),clims=(-ΔT/2,ΔT/2),aspect_ratio=1,c=:turbo)
        display(plot(p1))
        display(quiver!(Xp[:], Yp[:], quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black))
    end every nvis
    gif(anim, "../docs/porous_convection_2D.gif"; fps = 10)
end 

porous_convection_2_dim()
