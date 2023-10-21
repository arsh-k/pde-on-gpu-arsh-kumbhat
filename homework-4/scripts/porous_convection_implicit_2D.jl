using Plots, Plots.Measures, Printf
default(size=(1200, 800), framestyle=:box, label=false, grid=false, margin=10mm, lw=6, labelfontsize=20, tickfontsize=20, titlefontsize=24)

@views function porous_convection_2_dim_implicit()
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
    re_D      = 4π
    nt        = 500
    nvis      = 5
    # derived numerics
    dx        = lx / nx
    dy        = ly / ny
    xc        = LinRange(-lx/2 + dx / 2, lx/2 - dx / 2, nx)
    yc        = LinRange(-ly + dy / 2, -dy/2, ny)
    θ_dτ_D    = max(lx, ly)/re_D/cfl/min(dx, dy)
    β_dτ_D    = (re_D*k_ηf)/(cfl*min(dx, dy)*max(lx, ly))    
    dt_diff   = min(dx,dy)^2/λ_ρCp/4.1
    # array initialisation
    Pf        = @. exp(-(xc-lx/4)^2 -(yc'-ly/4)^2)
    qDx       = zeros(Float64, nx + 1, ny)
    qDy       = zeros(Float64, nx, ny+1) #Endpoints are zero fluxes.
    r_Pf      = zeros(Float64, nx, ny)
    r_T       = zeros(Float64, nx-2, ny-2)
    T         = @. ΔT*exp(-xc^2 - (yc'+ly/2)^2)
    T_old     = copy(T)
    T[:,1] .= ΔT/2; T[:,end] .= -ΔT/2
    T[[1,end],:] .= T[[2,end-1],:]
    dT_dt     = zeros(Float64, nx-2, ny-2)
    qTx       = zeros(Float64, nx+1, ny) #Fluxes are zero at the boundaries in the x-direction.
    qTy       = zeros(Float64, nx, ny-1) #Constant temperatures at the y-direction boundaries.
    # visualisation init
    st        = ceil(Int,nx/25)
    Xc, Yc    = [x for x=xc, y=yc], [y for x=xc,y=yc]
    Xp, Yp    = Xc[1:st:end,1:st:end], Yc[1:st:end,1:st:end]
    qDxc      = zeros(Float64, nx, ny)
    qDyc      = zeros(Float64, nx, ny)
    qDmag     = zeros(Float64, nx, ny)
    # physical time loop
    anim = @animate for it = 1:nt
        T_old .= (T)
        iter = 1; err_D = 2ϵtol; err_T = 2ϵtol;
        dt = if it == 1
            0.1*min(dx,dy)/(αρg*ΔT*k_ηf)
        else
            min(5.0*min(dx,dy)/(αρg*ΔT*k_ηf),ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1)
        end
        re_T    = π + sqrt(π^2 + ly^2/λ_ρCp/dt)
        θ_dτ_T  = max(lx,ly)/re_T/cfl/min(dx,dy)
        β_dτ_T  = (re_T*λ_ρCp)/(cfl*min(dx,dy)*max(lx,ly))
        # iteration loop 
        while max(err_D, err_T) >= ϵtol && iter <= maxiter
            # Darcy Flux Update
            qDx[2:end-1, :]         .-=  (qDx[2:end-1, :] .+ k_ηf .* diff(Pf, dims = 1) ./ dx)./ (1.0 .+ θ_dτ_D)
            qDy[:, 2:end-1]         .-=  (qDy[:, 2:end-1] .+ k_ηf .* diff(Pf, dims = 2) ./ dy)./ (1.0 .+ θ_dτ_D)
            #Boussinesq approximation
            qDx[2:end-1,:]           .= qDx[2:end-1,:] .+ (k_ηf * αρgx .* (T[1:end-1,:] + T[2:end,:] ./2) ./ (1.0 .+ θ_dτ_D))
            qDy[:,2:end-1]           .= qDy[:,2:end-1] .+ (k_ηf * αρgy .* (T[:,1:end-1] + T[:,2:end] ./2)  ./ (1.0 .+ θ_dτ_D))
            Pf                      .-=  (diff(qDx, dims = 1) ./ dx) ./β_dτ_D + (diff(qDy, dims = 2) ./ dy) ./ β_dτ_D
            #Temperature Update 
            dt_adv                    = ϕ*min(dx/maximum(abs.(qDx)), dy/maximum(abs.(qDy)))/2.1
            dt                        = min(dt_diff,dt_adv)
            qTx[2:end-1,:]           .= (θ_dτ_T.*qTx[2:end-1,:].- λ_ρCp.*diff(T, dims = 1) ./dx) ./ (1+θ_dτ_T)
            qTy                      .= (θ_dτ_T.*qTy .- λ_ρCp.*diff(T, dims = 2) ./dy) ./ (1+θ_dτ_T) 
            dT_dt                    .= (T[2:end-1, 2:end-1] .- T_old[2:end-1,2:end-1]) ./dt
            dT_dt                   .+= max.(qDx[2:end-2, 2:end-1],0.0) .* diff(T[1:end-1, 2:end-1], dims = 1)./dx ./ ϕ
            dT_dt                   .+= min.(qDx[3:end-1, 2:end-1],0.0) .* diff(T[2:end, 2:end-1], dims = 1)./dx ./ ϕ
            dT_dt                   .+= max.(qDy[2:end-1, 2:end-2],0.0) .* diff(T[2:end-1, 1:end-1], dims = 2)./dy ./ ϕ
            dT_dt                   .+= min.(qDy[2:end-1, 3:end-1],0.0) .* diff(T[2:end-1, 2:end], dims = 2)./dy ./ ϕ
            T[2:end-1,2:end-1]      .-= ( dT_dt .+ (diff(qTx[2:end-1,2:end-1], dims =1)./dx .+ diff(qTy[2:end-1,:], dims =2)./dy))./((1.0/dt) + β_dτ_T)
            T[[1,end],:] .= T[[2,end-1],:]
            if iter % ncheck == 0
                r_Pf  .=  diff(qDx, dims = 1)./dx .+ diff(qDy, dims = 2) ./dy
                r_T   .=  dT_dt .+ diff(qTx[2:end-1,2:end-1], dims = 1) ./dx .+ diff(qTy[2:end-1,:], dims = 2) ./dy
                err_D  = maximum(abs.(r_Pf))
                err_T  = maximum(abs.(r_T))
                @printf("it = %d, iter/nx=%.1f, err_Pf=%1.3e, err_T=%1.3e\n",it,iter/nx,err_D,err_T)
            end
            iter += 1
        end
        # visualisation
        qDxc    .= 0.5 .* (qDx[1:end-1,:] .+ qDx[2:end,:])
        qDyc    .= 0.5 .* (qDy[:,1:end-1] .+ qDy[:,2:end])
        qDmag   .= sqrt.(qDxc.^2 .+ qDyc.^2)
        qDxc    ./= qDmag
        qDyc    ./= qDmag
        qDx_p    = qDxc[1:st:end,1:st:end]
        qDy_p    = qDyc[1:st:end,1:st:end]
        p1       = heatmap(xc,yc,T';xlims=(xc[1],xc[end]),ylims=(yc[1],yc[end]),clims=(-ΔT/2,ΔT/2),aspect_ratio=1,c=:turbo, title = "Rayleigh number = $(Ra), it = $(it)")
        display(plot(p1))
        display(quiver!(Xp[:], Yp[:], quiver=(qDx_p[:], qDy_p[:]), lw=0.5, c=:black))   
    end every nvis
    gif(anim, "../docs/porous_convection_2D_implicit.gif"; fps = 10)
end

porous_convection_2_dim_implicit()
