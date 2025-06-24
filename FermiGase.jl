using Distributed
nprocs()<6 && addprocs(6-nprocs())
using SharedArrays

@everywhere include("src/SpinHall.jl")
@everywhere begin
    using LinearAlgebra
    myid()!=1 && BLAS.set_num_threads(1)
    using .SpinHall
    using ChunkSplitters
end

using Roots
using FileIO,JLD2
using CairoMakie,Printf,LaTeXStrings

set_theme!(size=(600,480), fonts=(regular="Times New Roman",bold="Times New Roman"))
const cm = 72/2.54
c = repeat(Makie.wong_colors(),3)


function set_lattice(v0, m0, g1, g2=g1*0.99537)
    mz = 0.0
    Kmax = 7
    b = [[1.0,1.0] [-1.0,1.0]]
    Lattice(b,v0,m0,mz,g1,g2,Kmax)
end

include("src/onsager_fermi.jl")


##
lat = set_lattice(4.0, 2.0, 0.25)
Γ = [0.0,0.0]
kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],64)
en = eigband(lat,kl.k, 1:12)

xt = (kl.r[kl.pt],["Γ","X","M","Γ"])
title = @sprintf("v_0=%.1f,\\; m_0=%.1f", lat.v0,lat.m0)|>latexstring
series(kl.r,en.-en[1]; axis=(title=title,xticks=xt), color=c)



##
kmesh = mymesh([-0.5.*(lat.b[:,1].+lat.b[:,2]), lat.b[:,1], lat.b[:,2]],[256,256])
q = [[1e-4,0] [-1e-4,0] [0.0,0]]
w = range(-3,3,256)
η = 0.1

mu = (en[4,1]+en[5,1])/2
t = time()
X = FermiGase(kmesh,lat,q,w,mu,η)
println("time_used:",time()-t)

save("data/X_n4_256_M20.jld2","mu_n4",mu,"X_n4",X,"q",q,"w",w,"eta",η,"v0",lat.v0,"m0",lat.m0,"Nx",size(kmesh,3))


## 
tmp = load("data/X_n4_256_M20.jld2")
w = tmp["w"]
X = tmp["X_n4"]
q = tmp["q"]
η = tmp["eta"]
ρu = 4

##
NS = ρu*lat.Sunit
sc_xy = (X.szjy[:,1].-X.szjy[:,2]).*(1im/(q[1,1]-q[1,2])/NS)
cs_yx = (X.jysz[:,1].-X.jysz[:,2]).*(1im/(q[1,1]-q[1,2])/NS)
sc_xx = (X.szjx[:,1].-X.szjx[:,2]).*(1im/(q[1,1]-q[1,2])/NS)
cs_xx = (X.jxsz[:,1].-X.jxsz[:,2]).*(1im/(q[1,1]-q[1,2])/NS)

begin
    title = @sprintf("(v_0,m_0,g_{11},g_{12},\\eta,\\rho_u)=(%.1f,%.1f,%.2f,%.4f,%.2f,%.1f)",lat.v0,lat.m0,lat.g1,lat.g2,η,ρu)|>latexstring
    f = Figure(size=(9,8).*80)

    ax1 = Axis(f[1,1],limits=((-1,1).*3.2,(-0.5,1).*0.06),title=title)
    scatterlines!(w,real.(sc_xy),label=L"\sigma^{sc}_{xy}",marker=:utriangle,markersize=6)
    scatterlines!(w,real.(cs_yx).*(-1),label=L"\sigma^{cs}_{yx}",markersize=6)
    axislegend(position=:ct,nbanks=2)

    ax2 = Axis(f[1,2],limits=((-1,1).*3.2,(-1,1).*0.06),title=title)
    scatterlines!(w,imag.(sc_xy),label=L"\sigma^{sc}_{xy}",marker=:utriangle,markersize=6)
    scatterlines!(w,imag.(cs_yx).*(-1),label=L"\sigma^{cs}_{yx}",markersize=6)
    axislegend(position=:ct,nbanks=2)

    ax3 = Axis(f[2,1],limits=(nothing,(-1,1).*0.08),title=title)
    scatterlines!(w,real.(sc_xx),label=L"\sigma^{sc}_{xx}",marker=:utriangle,markersize=6)
    scatterlines!(w,real.(cs_xx).*(1),label=L"\sigma^{cs}_{xx}",markersize=6)
    axislegend(position=:ct,nbanks=2)

    ax4 = Axis(f[2,2],limits=(nothing,(-1,1).*0.03),title=title)
    scatterlines!(w,imag.(sc_xx),label=L"\sigma^{sc}_{xx}",marker=:utriangle,markersize=6)
    scatterlines!(w,imag.(cs_xx).*(1),label=L"\sigma^{cs}_{xx}",markersize=6)
    axislegend(position=:ct,nbanks=2)
    f
end


## save to plot
save("data/FermiGase_inslator_M20.hdf5",Dict(
    "v0"=>lat.v0,"m0"=>lat.m0,"mz"=>lat.mz,"eta"=>η,
    "w"=>[w;],
    "sc_xy_re" => real.(sc_xy),
    "sc_xy_im"=> imag.(sc_xy), 
    "cs_yx_re" => real.(cs_yx),
    "cs_yx_im"=>imag.(cs_yx),

    "sc_xx_re" => real.(sc_xx),
    "sc_xx_im"=> imag.(sc_xx), 
    "cs_xx_re" => real.(cs_xx),
    "cs_xx_im"=>imag.(cs_xx))
)




## ------------- finite temperature ----------
kmesh = mymesh([-0.5.*(lat.b[:,1].+lat.b[:,2]), lat.b[:,1], lat.b[:,2]],[256,256])
q = [[1e-4,0] [-1e-4,0] [0.0,0]]
w = range(-3,3,256)
η = 0.1
T = 0.1
ρu = 3

t = time()
mu = cal_mu(lat,kmesh,4,ρu,T)
println("mu:",mu,", time_used:",time()-t)
X = FermiGase(kmesh,lat,q,w,mu,η,T)
println("time_used:",time()-t)

save("data/X_n3_256_M20_T01.jld2","mu_n4",mu,"X_n4",X,"q",q,"w",w,"eta",η,"v0",lat.v0,"m0",lat.m0,"Nx",size(kmesh,3),"T",T)


## 
tmp = load("data/X_n3_256_M20_T01.jld2")
w = tmp["w"]
X = tmp["X_n4"]
q = tmp["q"]
η = tmp["eta"]
T = tmp["T"]
ρu = 3

##
NS = ρu*lat.Sunit
sc_xy = (X.szjy[:,1].-X.szjy[:,2]).*(1im/(q[1,1]-q[1,2])/NS)
cs_yx = (X.jysz[:,1].-X.jysz[:,2]).*(1im/(q[1,1]-q[1,2])/NS)
sc_xx = (X.szjx[:,1].-X.szjx[:,2]).*(1im/(q[1,1]-q[1,2])/NS)
cs_xx = (X.jxsz[:,1].-X.jxsz[:,2]).*(1im/(q[1,1]-q[1,2])/NS)

begin
    title = @sprintf("(v_0,m_0,g_{11},g_{12},\\eta,\\rho_u)=(%.1f,%.1f,%.2f,%.4f,%.2f,%.1f)",lat.v0,lat.m0,lat.g1,lat.g2,η,ρu)|>latexstring
    f = Figure(size=(9,8).*80)

    ax1 = Axis(f[1,1],limits=((-1,1).*3.2,(-0.5,1).*0.06),title=title)
    scatterlines!(w,real.(sc_xy),label=L"Re$\sigma^{sc}_{xy}$",marker=:utriangle,markersize=6)
    scatterlines!(w,real.(cs_yx).*(-1),label=L"Re$\sigma^{cs}_{yx}$",markersize=6)
    axislegend(position=:ct,nbanks=2)

    ax2 = Axis(f[1,2],limits=((-1,1).*3.2,(-1,1).*0.06),title=title)
    scatterlines!(w,imag.(sc_xy),label=L"Im$\sigma^{sc}_{xy}$",marker=:utriangle,markersize=6)
    scatterlines!(w,imag.(cs_yx).*(-1),label=L"Im$\sigma^{cs}_{yx}$",markersize=6)
    axislegend(position=:ct,nbanks=2)

    ax3 = Axis(f[2,1],limits=(nothing,(-1,1).*0.08),title=title)
    scatterlines!(w,real.(sc_xx),label=L"Re$\sigma^{sc}_{xx}$",marker=:utriangle,markersize=6)
    scatterlines!(w,real.(cs_xx).*(1),label=L"Re$\sigma^{cs}_{xx}$",markersize=6)
    axislegend(position=:ct,nbanks=2)

    ax4 = Axis(f[2,2],limits=(nothing,(-1,1).*0.03),title=title)
    scatterlines!(w,imag.(sc_xx),label=L"Im$\sigma^{sc}_{xx}$",marker=:utriangle,markersize=6)
    scatterlines!(w,imag.(cs_xx).*(1),label=L"Im$\sigma^{cs}_{xx}$",markersize=6)
    axislegend(position=:ct,nbanks=2)
    f
end


## save to plot
save("data/FermiGase_T.hdf5",Dict(
    "v0"=>lat.v0,"m0"=>lat.m0,"mz"=>lat.mz,"eta"=>η,
    "T"=>T,
    "w"=>[w;],
    "sc_xy_re" => real.(sc_xy),
    "sc_xy_im"=> imag.(sc_xy), 
    "cs_yx_re" => real.(cs_yx),
    "cs_yx_im"=>imag.(cs_yx),

    "sc_xx_re" => real.(sc_xx),
    "sc_xx_im"=> imag.(sc_xx), 
    "cs_xx_re" => real.(cs_xx),
    "cs_xx_im"=>imag.(cs_xx))
)











## ------------- check -------------
kmesh = mymesh([-0.5.*(lat.b[:,1].+lat.b[:,2]), lat.b[:,1], lat.b[:,2]],[512,512])
w = range(-3,3,256)
η = 0.1
T = 0.05
ρu= 1
t = time()
mu = cal_mu(lat,kmesh,4,ρu,T)
println("mu:",time()-t)

t = time()
X = FermiHall(kmesh,lat,w,mu,η,T)
println("time_used:",time()-t)

##

begin
    f,_,hm = heatmap(imag.(X.jxjy[129,:,:]),axis=(aspect=1,),colormap=:bwr)
    Colorbar(f[1,2],hm)
    f
end

Xxx = dropdims(sum(X.jxjx,dims=(2,3)),dims=(2,3))./(size(kmesh,2)*size(kmesh,3))
σxx.= (Xxx.+2).*(1im)
begin
    title = @sprintf("(v_0,m_0,g_{11},g_{12},\\eta,\\rho_u)=(%.1f,%.1f,%.2f,%.4f,%.2f,%.1f)",lat.v0,lat.m0,lat.g1,lat.g2,η,ρu)|>latexstring
    f = Figure(size=(9,4).*80)

    ax1 = Axis(f[1,1],limits=(nothing,(-0.1,1).*0.1),title=title)
    scatterlines!(w,real.(Xxx),label=L"\sigma^{sc}_{xy}",marker=:utriangle,markersize=6)

    ax2 = Axis(f[1,2],limits=(nothing,(-1,1).*100),title=title)
    scatterlines!(w,imag.(σxx),label=L"\sigma^{sc}_{xy}",marker=:utriangle,markersize=6)

    f
end
scatterlines(w,imag.(Xxx))