using LinearAlgebra
using Revise
include("src/SpinHall.jl")
using .SpinHall

using OrderedCollections,FileIO
using CairoMakie
using Printf,LaTeXStrings

set_theme!(size=(600,480), fonts=(regular="Times New Roman",bold="Times New Roman"))
const cm = 72/2.54
c = repeat(Makie.wong_colors(),3)

function set_lattice(v0, m0, g1, g2=g1*0.2594/0.26;mz=0.0)
    Kmax = 7
    b = [[1.0,1.0] [-1.0,1.0]]
    Lattice(b,v0,m0,mz,g1,g2,Kmax)
end

## -------------------------------------------------------------
#                   单粒子能谱
#  -------------------------------------------------------------
#                (V0, M0,  g_11)
lat = set_lattice(4.0,2.0, 0.25,0.1;mz=0.0)
Γ = [0.0,0.0]
kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],128)
en = eigband(lat,kl.k, 1:12)

xt = (kl.r[kl.pt],["Γ","X","M","Γ"])
title = @sprintf("v_0=%.1f,\\; m_0=%.1f", lat.v0,lat.m0)|>latexstring
series(kl.r,en.-en[1]; axis=(title=title,xticks=xt), color=c)

##

function calJu!(lat,k,μ)
    (;NK, Kvec) = lat
    Ju = Array{Float64}(undef,2*NK)
    @inbounds for i in 1:NK
        Ju[i] = Ju[i+NK] = 2*(k[μ]+Kvec[μ,i])
    end
    return Ju
end

fE(en,mu) = en<mu ? 1 : 0
fE(en,mu,β) = 1/(exp(β*(en-mu))+1)+1e-16

function sigma1(en,ev,jx,jy,sz,mu)
    jxm = Array{ComplexF64}(undef,length(en))
    jym = similar(jxm)
    sn = similar(jxm)
    s = 0.0im
    @views for m in eachindex(en)
        fE(en[m],mu) == 0 && continue
        jxm .= jx.*ev[:,m]
        jym .= jy.*ev[:,m]
        for n in eachindex(en)
            Enm = en[n]-en[m]
            abs(Enm)<1e-8 && continue
            Rmn = dot(jxm,ev[:,n])/Enm
            sn .= sz.*ev[:,n]
            for n′ in eachindex(en)
                abs(en[n]-en[n′])>1e-8 && continue
                Emn′ = en[m]-en[n′]
                abs(Emn′)<1e-8 && continue
                Rn′m = dot(ev[:,n′],jym)/Emn′
                s+= Rmn*Rn′m*dot(sn,ev[:,n′])
            end
        end
    end
    return 2im*s
end

function mysum(v1::AbstractVector{T},v2::AbstractVector{T}) where {T<:Number}
    s = zero(T)
    @inbounds for i in eachindex(v1)
        s+= v1[i]*v2[i]
    end
    return s
end

function fermiHall(
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

    X_jxjx = zeros(ComplexF64,length(w),2)
    X_szjx = zeros(ComplexF64,length(w),2)
    X_jxsz = zeros(ComplexF64,length(w),2)
    X_szjy = zeros(ComplexF64,length(w),2)
    X_jysz = zeros(ComplexF64,length(w),2)
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
    return (;X_jxjx,X_szjx,X_jxsz,X_szjy,X_jysz)
end

##
k = [0.2,0.0]
jx = calJu!(lat,k,1)
jy = calJu!(lat,k,2)
sz = [ones(lat.NK); fill(-1.0,lat.NK)]
ek,vk = eigenband(lat,k,1:2*lat.NK)

##
mu = (en[4,1]+en[5,1])/2
sigma1(ek,vk,jx,jy,sz,mu)

##
q = [[5e-4,0] [-5e-4,0] [0.0,0]]
w = range(-1,1,64)
ek2,vk2 = eigenband(lat,q.+k,1:2*lat.NK)
jx2 = [calJu!(lat,k.+0.5.*q[:,1],1) calJu!(lat,k.+0.5.*q[:,2],1)]
jy2 = [calJu!(lat,k.+0.5.*q[:,1],2) calJu!(lat,k.+0.5.*q[:,2],2)]
X = fermiHall(ek2,vk2,sz,jx2,jy2,w,mu,0.0)

sc = (X.X_szjy[:,1].-X.X_szjy[:,2]).*(1im/(q[1,1]-q[1,2]))
sc[32]