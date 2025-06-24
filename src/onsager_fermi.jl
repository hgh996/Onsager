@everywhere begin

function mysum(v1::AbstractVector{T},v2::AbstractVector{T}) where {T<:Number}
    s = zero(T)
    @inbounds for i in eachindex(v1)
        s+= v1[i]*v2[i]
    end
    return s
end

fE(en,mu) = en<mu ? 1 : 0
fE(en,mu,β) = 1/(exp(β*(en-mu))+1)

# 零温
function fermiHall!(
    X_jxjx::AbstractArray{ComplexF64,2},
    X_szjx::AbstractArray{ComplexF64,2},
    X_jxsz::AbstractArray{ComplexF64,2},
    X_szjy::AbstractArray{ComplexF64,2},
    X_jysz::AbstractArray{ComplexF64,2},

    en::AbstractArray{Float64,2},
    ev::AbstractArray{<:Number,3},

    sz::AbstractVector{<:Number},
    jx::AbstractArray{<:Number,2},
    jy::AbstractArray{<:Number,2},

    w::AbstractVector{Float64},
    μ::Float64,
    η::Float64
)
    Sz′ = Array{ComplexF64}(undef,size(ev,1))
    Jx′ = Array{ComplexF64}(undef,size(ev,1),2)
    Jy′ = similar(Jx′)
    @views for n in axes(en,1)
        Ek = en[n,3]
        fn = fE(Ek,μ)
        Sz′.= conj.(sz.*ev[:,n,3])
        Jx′[:,1].= conj.(jx[:,1].*ev[:,n,3])
        Jx′[:,2].= conj.(jx[:,2].*ev[:,n,3])
        Jy′[:,1].= conj.(jy[:,1].*ev[:,n,3])
        Jy′[:,2].= conj.(jy[:,2].*ev[:,n,3])

        for iq in 1:2
            for m in axes(en,1)
                fm = fE(en[m,iq],μ)
                dF = fn-fm
                dF == 0 && continue
                
                Sz_nm = mysum(Sz′,ev[:,m,iq])
                Jx_nm = mysum(Jx′[:,iq],ev[:,m,iq])
                Jy_nm = mysum(Jy′[:,iq],ev[:,m,iq])

                jxjx = Jx_nm*conj(Jx_nm)
                szjx = Sz_nm*conj(Jx_nm)
                jxsz = conj(szjx)
                szjy = Sz_nm*conj(Jy_nm)
                jysz = conj(szjy)
                
                dE = Ek-en[m,iq]+1im*η
                for iw in eachindex(w)
                    X_jxjx[iw,iq] += jxjx*dF/(w[iw]+dE)
                    X_szjx[iw,iq] += szjx*dF/(w[iw]+dE)
                    X_jxsz[iw,iq] += jxsz*dF/(w[iw]+dE)
                    X_szjy[iw,iq] += szjy*dF/(w[iw]+dE)
                    X_jysz[iw,iq] += jysz*dF/(w[iw]+dE)
                end
            end
        end
    end
    nothing
end

# 有限温度
function fermiHall!(
    X_jxjx::AbstractArray{ComplexF64,2},
    X_szjx::AbstractArray{ComplexF64,2},
    X_jxsz::AbstractArray{ComplexF64,2},
    X_szjy::AbstractArray{ComplexF64,2},
    X_jysz::AbstractArray{ComplexF64,2},

    en::AbstractArray{Float64,2},
    ev::AbstractArray{<:Number,3},

    sz::AbstractVector{<:Number},
    jx::AbstractArray{<:Number,2},
    jy::AbstractArray{<:Number,2},

    w::AbstractVector{Float64},
    μ::Float64,
    η::Float64,
    β::Float64
)
    Sz′ = Array{ComplexF64}(undef,size(ev,1))
    Jx′ = Array{ComplexF64}(undef,size(ev,1),2)
    Jy′ = similar(Jx′)
    @views for n in axes(en,1)
        Ek = en[n,3]
        fn = fE(Ek,μ,β)
        Sz′.= conj.(sz.*ev[:,n,3])
        Jx′[:,1].= conj.(jx[:,1].*ev[:,n,3])
        Jx′[:,2].= conj.(jx[:,2].*ev[:,n,3])
        Jy′[:,1].= conj.(jy[:,1].*ev[:,n,3])
        Jy′[:,2].= conj.(jy[:,2].*ev[:,n,3])

        for iq in 1:2
            for m in axes(en,1)
                fm = fE(en[m,iq],μ,β)
                dF = fn-fm
                abs(dF) < 1e-9 && continue
                
                Sz_nm = mysum(Sz′,ev[:,m,iq])
                Jx_nm = mysum(Jx′[:,iq],ev[:,m,iq])
                Jy_nm = mysum(Jy′[:,iq],ev[:,m,iq])

                jxjx = Jx_nm*conj(Jx_nm)
                szjx = Sz_nm*conj(Jx_nm)
                jxsz = conj(szjx)
                szjy = Sz_nm*conj(Jy_nm)
                jysz = conj(szjy)
                
                dE = Ek-en[m,iq]+1im*η
                for iw in eachindex(w)
                    X_jxjx[iw,iq] += jxjx*dF/(w[iw]+dE)
                    X_szjx[iw,iq] += szjx*dF/(w[iw]+dE)
                    X_jxsz[iw,iq] += jxsz*dF/(w[iw]+dE)
                    X_szjy[iw,iq] += szjy*dF/(w[iw]+dE)
                    X_jysz[iw,iq] += jysz*dF/(w[iw]+dE)
                end
            end
        end
    end
    nothing
end

function calDhk!(hk,k,Kvec,μ,NK)
    @inbounds for i in axes(Kvec,2)
        kk = 2*(k[μ]+Kvec[μ,i])
        hk[i] = kk
        hk[i+NK] = kk
    end
    nothing
end

end

# 零温
function FermiGase(kmesh,lat,q,w,μ,η)
    nth = nworkers()
    _,Nx,Ny = size(kmesh)
    (;v0,m0,mz,NK,Kcoe,Kvec) = lat

    sz = [fill(0.5,NK); fill(-0.5,NK)]
    ky_split = chunks(1:Ny, n=nth)

    X_jxjx = SharedArray{ComplexF64}(length(w),2,nth)
    X_szjx = SharedArray{ComplexF64}(length(w),2,nth)
    X_jxsz = SharedArray{ComplexF64}(length(w),2,nth)
    X_szjy = SharedArray{ComplexF64}(length(w),2,nth)
    X_jysz = SharedArray{ComplexF64}(length(w),2,nth)

    @sync @distributed for i in 1:nth
        mat = zeros(ComplexF64,2*NK,2*NK)
        SpinHall.matoff!(mat,v0,m0,Kcoe)
        en = Array{Float64}(undef,2*NK,3)
        ev = Array{ComplexF64}(undef,2*NK,2*NK,3)
        k = Array{Float64}(undef,2)
        jx = Array{Float64}(undef,2*NK,2) # j_{q}, j_{-q}
        jy = similar(jx)

        @views for iy in ky_split[i],ix in 1:Nx
            for iq in 1:3
                k.= kmesh[:,ix,iy].+q[:,iq]
                SpinHall.matdiag!(mat,k,Kvec,v0,mz)
                _en,_ev = eigen(Hermitian(mat))
                en[:,iq].= _en
                ev[:,:,iq].= _ev

                iq == 3 && break
                k.= kmesh[:,ix,iy].+0.5.*q[:,iq]
                calDhk!(jx[:,iq],k,Kvec,1,NK)
                calDhk!(jy[:,iq],k,Kvec,2,NK)
            end

            fermiHall!(X_jxjx[:,:,i],X_szjx[:,:,i],X_jxsz[:,:,i],X_szjy[:,:,i],X_jysz[:,:,i],en,ev,sz,jx,jy,w,μ,η)
        end
    end
    jxjx = dropdims(sum(X_jxjx,dims=3),dims=3)./(Nx*Ny)
    szjx = dropdims(sum(X_szjx,dims=3),dims=3)./(Nx*Ny)
    jxsz = dropdims(sum(X_jxsz,dims=3),dims=3)./(Nx*Ny)
    szjy = dropdims(sum(X_szjy,dims=3),dims=3)./(Nx*Ny)
    jysz = dropdims(sum(X_jysz,dims=3),dims=3)./(Nx*Ny)

    return (;jxjx, szjx, jxsz, szjy, jysz)
end

# 有限温度
function FermiGase(kmesh,lat,q,w,μ,η,T)
    nth = nworkers()
    _,Nx,Ny = size(kmesh)
    (;v0,m0,mz,NK,Kcoe,Kvec) = lat
    β = 1/T

    sz = [fill(0.5,NK); fill(-0.5,NK)]
    ky_split = chunks(1:Ny, n=nth)

    X_jxjx = SharedArray{ComplexF64}(length(w),2,nth)
    X_szjx = SharedArray{ComplexF64}(length(w),2,nth)
    X_jxsz = SharedArray{ComplexF64}(length(w),2,nth)
    X_szjy = SharedArray{ComplexF64}(length(w),2,nth)
    X_jysz = SharedArray{ComplexF64}(length(w),2,nth)

    @sync @distributed for i in 1:nth
        mat = zeros(ComplexF64,2*NK,2*NK)
        SpinHall.matoff!(mat,v0,m0,Kcoe)
        en = Array{Float64}(undef,2*NK,3)
        ev = Array{ComplexF64}(undef,2*NK,2*NK,3)
        k = Array{Float64}(undef,2)
        jx = Array{Float64}(undef,2*NK,2) # j_{q}, j_{-q}
        jy = similar(jx)

        @views for iy in ky_split[i],ix in 1:Nx
            for iq in 1:3
                k.= kmesh[:,ix,iy].+q[:,iq]
                SpinHall.matdiag!(mat,k,Kvec,v0,mz)
                _en,_ev = eigen(Hermitian(mat))
                en[:,iq].= _en
                ev[:,:,iq].= _ev

                iq == 3 && break
                k.= kmesh[:,ix,iy].+0.5.*q[:,iq]
                calDhk!(jx[:,iq],k,Kvec,1,NK)
                calDhk!(jy[:,iq],k,Kvec,2,NK)
            end

            fermiHall!(X_jxjx[:,:,i],X_szjx[:,:,i],X_jxsz[:,:,i],X_szjy[:,:,i],X_jysz[:,:,i],en,ev,sz,jx,jy,w,μ,η,β)
        end
    end
    jxjx = dropdims(sum(X_jxjx,dims=3),dims=3)./(Nx*Ny)
    szjx = dropdims(sum(X_szjx,dims=3),dims=3)./(Nx*Ny)
    jxsz = dropdims(sum(X_jxsz,dims=3),dims=3)./(Nx*Ny)
    szjy = dropdims(sum(X_szjy,dims=3),dims=3)./(Nx*Ny)
    jysz = dropdims(sum(X_jysz,dims=3),dims=3)./(Nx*Ny)

    return (;jxjx, szjx, jxsz, szjy, jysz)
end

function cal_mu(lat,kmesh,Nb,n)
    _,Nx,Ny = size(kmesh)
    en = eigband(lat, kmesh, 1:Nb)
    Dos = sort(reshape(en,:))

    idx = Nx*Ny*n
    mu = (Dos[idx]+Dos[idx+1])/2
    return mu
end


function cal_Nu(en,mu,β)
    s = 0.0
    for i in en
        s+=fE(i,mu,β)
    end
    return s
end
function cal_mu(lat,kmesh,Nb,n,T)
    _,Nx,Ny = size(kmesh)
    en = eigband(lat, kmesh, 1:Nb)
    Dos = sort(reshape(en,:))

    idx = Nx*Ny*n
    β = 1/T
    foo(μ) = cal_Nu(Dos,μ,β)-idx

    mu = find_zero(foo, (Dos[1],Dos[end]), Roots.Brent())

    return mu
end



## -------------- 检查电导 --------------
@everywhere function fermiHall!(
    X_jxjx::AbstractArray{ComplexF64,1},
    X_jxjy::AbstractArray{ComplexF64,1},

    en::AbstractArray{Float64,1},
    ev::AbstractArray{<:Number,2},

    jx::AbstractArray{<:Number,1},
    jy::AbstractArray{<:Number,1},

    w::AbstractVector{Float64},
    μ::Float64,
    η::Float64,
    β::Float64
)
    Jx′ = Array{ComplexF64}(undef,size(ev,1))
    Jy′ = similar(Jx′)
    @views for n in axes(en,1)
        Ek = en[n]
        fn = fE(Ek,μ,β)
        Jx′.= conj.(jx.*ev[:,n])
        Jy′.= conj.(jy.*ev[:,n])

        for m in axes(en,1)
            fm = fE(en[m],μ,β)
            dF = fn-fm
            abs(dF) < 1e-9 && continue
  
            Jx_nm = mysum(Jx′,ev[:,m])
            Jy_nm = mysum(Jy′,ev[:,m])

            jxjx = Jx_nm*conj(Jx_nm)
            jxjy = Jx_nm*conj(Jy_nm)
            
            dE = Ek-en[m]+1im*η
            for iw in eachindex(w)
                X_jxjx[iw] += jxjx*dF/(w[iw]+dE)
                X_jxjy[iw] += jxjy*dF/(w[iw]+dE)
            end
        end
    end
    nothing
end

function FermiHall(kmesh,lat,w,μ,η,T)
    nth = nworkers()
    _,Nx,Ny = size(kmesh)
    (;v0,m0,mz,NK,Kcoe,Kvec) = lat
    β = 1/T

    ky_split = chunks(1:Ny, n=nth)

    X_jxjx = SharedArray{ComplexF64}(length(w),Nx,Ny,nth)
    X_jxjy = SharedArray{ComplexF64}(length(w),Nx,Ny,nth)
    @sync @distributed for i in 1:nth
        mat = zeros(ComplexF64,2*NK,2*NK)
        SpinHall.matoff!(mat,v0,m0,Kcoe)

        jx = Array{Float64}(undef,2*NK)
        jy = similar(jx)

        @views for iy in ky_split[i],ix in 1:Nx
            SpinHall.matdiag!(mat,kmesh[:,ix,iy],Kvec,v0,mz)
            en,ev = eigen(Hermitian(mat))

            calDhk!(jx,kmesh[:,ix,iy],Kvec,1,NK)
            calDhk!(jy,kmesh[:,ix,iy],Kvec,2,NK)

            fermiHall!(X_jxjx[:,ix,iy,i],X_jxjy[:,ix,iy,i],en,ev,jx,jy,w,μ,η,β)
        end
    end
    jxjx = dropdims(sum(X_jxjx,dims=4),dims=4)
    jxjy = dropdims(sum(X_jxjy,dims=4),dims=4)

    return (;jxjx, jxjy)
end