function cal_SC1_w(ϕG,w,lat,u0; θ1=0.5pi,θ2=0.0,η=0.0)
    q = [0.0,0.0]
    Mk0 = cal_BdG(lat,ϕG,u0,q)
    J1 = cal_Jθ(ϕG, q, lat.Kvec, θ1, sp=-1).*0.5
    J2 = cal_Jθ(ϕG, q, lat.Kvec, θ2, sp=1)
    X = Green1(Mk0,w,J1,J2,η=η)
    return X
end

function cal_SC1_qw(ϕG,q1,q2,w,lat,u0; θ1=0.5pi,θ2=0.0,η=0.0)
    X = Array{ComplexF64}(undef, length(w),length(q1),length(q2))
    for j in eachindex(q2), i in eachindex(q1)
        Mk0 = cal_BdG(lat,ϕG,u0,[q1[i],q2[j]])
        J1 = cal_Jθ(ϕG, [q1[i]/2,q2[j]/2], lat.Kvec, θ1, sp=-1)
        J2 = cal_Jθ(ϕG, [q1[i]/2,q2[j]/2], lat.Kvec, θ2, sp=1)
        X[:,i,j].= Green1(Mk0,w,J1,J2,η=η)
    end
    return X
end

function cal_SC2_w(ϕG,w,lat,u0,Js; θ1=0.5pi, θ2=0.0,η=0.0)
    longitudinal = abs(θ1-θ2)<1e-10
    @assert longitudinal || abs(abs(θ1-θ2)-0.5pi)<1e-10 "θ input error"
    
    Nm= lat.NK*2
    X = Array{ComplexF64}(undef, length(w))
    J2 = cal_Jθ(ϕG, [0.0,0.0], lat.Kvec, θ2,sp=1)
    DJ2= [ϕG; -1.0.*conj.(ϕG)]

    dH = cal_Dθ([0.0,0.0], lat.Kvec, θ1)
    for i in Nm+1:2Nm
        dH[i,i]*=-1
    end

    Gw = -1.0.*cal_BdG(lat,ϕG,u0,[0.0,0.0])
    Gii= diag(Gw)
    G = similar(Gw)
    for iw in eachindex(w)
        ww = abs(w[iw])>1e-4 ? w[iw]+1im*η : 1e-4+1im*η
        for iQ in 1:Nm
            Gw[iQ,iQ] = Gii[iQ]+ww
            Gw[iQ+Nm,iQ+Nm]= Gii[iQ+Nm] - ww
        end
        G.= inv(Gw)
        X[iw] = dot(Js,G*dH*G,J2)*(-1/ww)

        if longitudinal
            X[iw] += dot(Js,G,DJ2)*(-1/ww)
        end
    end

    return X
end

function cal_SC2_qw(ϕG,q1,q2,w,lat,u0,Js; θ2=0.0,η=0.0)
    X = Array{ComplexF64}(undef, length(w),length(q1),length(q2))
    for j in eachindex(q2), i in eachindex(q1)
        Mk0 = cal_BdG(lat,ϕG,u0,[q1[i],q2[j]])
        J2 = cal_Jθ(ϕG, [q1[i]/2,q2[j]/2], lat.Kvec, θ2,sp=1)
        X[:,i,j].= Green1(Mk0,w,Js,J2,η=η)
    end
    return X
end

function cal_SC_w(ϕG,w,lat,u0,Jsz; θ1=0.0,θ2=0.5pi,η=0.0)
    longitudinal = abs(θ1-θ2)<1e-10
    @assert longitudinal || abs(abs(θ1-θ2)-0.5pi)<1e-10 "θ input error"

    Nm= lat.NK*2
    X = Array{ComplexF64}(undef, length(w))

    q = [0.0,0.0]
    J2 = cal_Jθ(ϕG, q, lat.Kvec, θ2,sp=1)
    DJ2= [ϕG; (-1).*conj.(ϕG)]

    dH = cal_Dθ(q, lat.Kvec, θ1)
    for i in Nm+1:2Nm
        dH[i,i]*=-1
    end

    Gw = -1.0.*cal_BdG(lat,ϕG,u0,q)
    Gii= diag(Gw)
    G = similar(Gw)
    for iw in eachindex(w)
        ww = w[iw]+1im*η
        for iQ in 1:Nm
            Gw[iQ,iQ] = Gii[iQ]+ww
            Gw[iQ+Nm,iQ+Nm]= Gii[iQ+Nm] - ww
        end
        G.= inv(Gw)

        X[iw] = dot(Jsz,G*dH*G,J2)*1im
        if longitudinal
            X[iw] += dot(Jsz,G,DJ2)*1im
        end
    end
    
    return X
end

function cal_SC_qw(ϕG,q1,q2,w,lat,u0,Jsz; θ2=0.5pi,η=0.0)
    X = Array{ComplexF64}(undef, length(w),length(q1),length(q2))
    for j in eachindex(q2), i in eachindex(q1)
        Mk0 = cal_BdG(lat,ϕG,u0,[q1[i],q2[j]])
        J2 = cal_Jθ(ϕG, [q1[i]/2,q2[j]/2], lat.Kvec, θ2,sp=1)
        X[:,i,j].= SpinHall.GreenX(Mk0,w,Jsz,J2,η=η)
    end
    return X
end


function cal_CS_qw(ϕG,q1,q2,w,lat,u0,Jsz; θ1=0.0,η=0.0)
    X = Array{ComplexF64}(undef, length(w),length(q1),length(q2))
    for j in eachindex(q2), i in eachindex(q1)
        Mk0 = cal_BdG(lat,ϕG,u0,[q1[i],q2[j]])
        J1 = cal_Jθ(ϕG, [q1[i]/2,q2[j]/2], lat.Kvec, θ1,sp=1)
        X[:,i,j].= SpinHall.GreenX(Mk0,w,J1,Jsz,η=η)
    end
    return X
end

function cal_CS_w(ϕG,w,lat,u0,Jsz; θ1=0.0,θ2=0.5pi,η=0.0)
    longitudinal = abs(θ1-θ2)<1e-10
    @assert longitudinal || abs(abs(θ1-θ2)-0.5pi)<1e-10 "θ input error"

    Nm= lat.NK*2
    X = Array{ComplexF64}(undef, length(w))

    q = [0.0,0.0]
    J1 = cal_Jθ(ϕG, q, lat.Kvec, θ1,sp=1)
    DJ1= [ϕG; (-1).*conj.(ϕG)]

    dH = cal_Dθ(q, lat.Kvec, θ2)
    for i in Nm+1:2Nm
        dH[i,i]*=-1
    end

    Gw = -1.0.*cal_BdG(lat,ϕG,u0,q)
    Gii= diag(Gw)
    G = similar(Gw)
    for iw in eachindex(w)
        ww = w[iw]+1im*η
        for iQ in 1:Nm
            Gw[iQ,iQ] = Gii[iQ]+ww
            Gw[iQ+Nm,iQ+Nm]= Gii[iQ+Nm] - ww
        end
        G.= inv(Gw)

        X[iw] = dot(J1,G*dH*G,Jsz)*1im
        if longitudinal
            X[iw] += dot(DJ1,G,Jsz)*1im
        end
    end

    return X
end


function cal_stz(ψup,ψdn,x,y)
    sx = Array{Float64}(undef,length(x),length(y))
    sy = similar(sx)
    for iy in eachindex(y),ix in eachindex(x)
        Ω = -sin(y[iy])cos(x[ix])-1im*sin(x[ix])cos(y[iy])
        tmp = conj(ψup[ix,iy])*Ω*ψdn[ix,iy]
        sx[ix,iy] = real(tmp)*x[ix]
        sy[ix,iy] = real(tmp)*y[iy]
    end
    return sx,sy
end

function cal_stz(ψup,ψdn,xmesh)
    sx = Array{Float64}(undef,size(xmesh,2),size(xmesh,3))
    sy = similar(sx)
    @views for iy in axes(sx,2),ix in axes(sx,1)
        x,y = xmesh[:,ix,iy]
        Ω = -sin(y)cos(x)-1im*sin(x)cos(y)
        tmp = conj(ψup[ix,iy])*Ω*ψdn[ix,iy]
        sx[ix,iy] = real(tmp)
        sy[ix,iy] = real(tmp)
    end
    return sx,sy
end


function cell_ave(arr::Array{T,2},x,y,Nx,Ny) where {T<:Number}
    NNx = div(length(x),Nx)
    NNy = div(length(y),Ny)
    Su = (x[NNx]-x[1])*(y[NNy]-y[1])/(NNx*NNy)
    s = Array{T}(undef,Nx,Ny)
    for iy in 1:Ny,ix in 1:Nx
        i = (ix-1)*NNx+1
        j = (iy-1)*NNy+1
        s[ix,iy] = sum(arr[i:i+NNx-1,j:j+NNy-1])*Su
    end
    return s
end