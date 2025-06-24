using LinearAlgebra
using Revise
include("src/SpinHall.jl")
using .SpinHall
includet("src/onsager_bose.jl")

using FileIO
using CairoMakie
using Printf,LaTeXStrings

set_theme!(size=(600,480), fonts=(regular="Times New Roman",bold="Times New Roman"))
const cm = 72/2.54
c = repeat(Makie.wong_colors(),3)

function set_lattice(v0, m0, g1, g2=g1*0.99537; mz=0.0)
    Kmax = 7
    b = [[1.0,1.0] [-1.0,1.0]]
    Lattice(b,v0,m0,mz,g1,g2,Kmax)
end


## -------------------------------------------------------------
#                   单粒子能谱
#  -------------------------------------------------------------
#                (V0, M0,  g_11)
lat = set_lattice(4.0,2.5, 0.25;mz=0.0)
Γ = [0.0,0.0]
kl = BzLine([Γ, 0.5.*lat.b[:,1], 0.5.*(lat.b[:,1].+lat.b[:,2]), Γ],128)
en = eigband(lat,kl.k, 1:12)

xt = (kl.r[kl.pt],["Γ","X","M","Γ"])
title = @sprintf("v_0=%.1f,\\; m_0=%.1f", lat.v0,lat.m0)|>latexstring
series(kl.r,en.-en[1]; axis=(title=title,xticks=xt), color=c)



## ---------------------------------------------------
#               mean field Ground state
# ---------------------------------------------------
Nopt = 10
E0,ϕ0= eigenband(lat, Γ, 1:Nopt)
init_gs = ComplexF64[0.0,1.0, zeros(Nopt-2)...] # perpendicular phase
# init_gs = ComplexF64[1.0,cis(-0.25), zeros(Nopt-2)...] # in-plane phase

ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=init_gs, Nstep=10^5)
mat = calmat(lat, Γ)
ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=10^4)
SpinHall.gaugephi0!(ϕG, ϕ0)
dot_spin(ϕG)



## --- plot wave function ---
xx = range(-pi,pi,128)
ψup = cal_bloch_wave(Γ,ϕG[1:lat.NK],lat,xx,xx)
ψdn = cal_bloch_wave(Γ,ϕG[lat.NK+1:end],lat,xx,xx)

title = @sprintf("(V_0,M_0,g_{11},g_{12},m_z)=(%.1f,%.1f,%.2f,%.4f,%.2f)",lat.v0,lat.m0,lat.g1,lat.g2,lat.mz)|>latexstring
fig=Figure(size=(650,600))
_,hm1 = heatmap(fig[1,1],xx,xx,abs2.(ψup),axis=(aspect=1,title=L"|\phi_{\uparrow}|^2"),colormap=:jet)
Colorbar(fig[1,2],hm1)
_,hm2 = heatmap(fig[1,3],xx,xx,angle.(ψup),axis=(aspect=1,title=L"arg$\phi_{\uparrow}$"),colormap=:jet)
Colorbar(fig[1,4],hm2)

_,hm3 = heatmap(fig[2,1],xx,xx,abs2.(ψdn),axis=(aspect=1,title=L"|\phi_{\downarrow}|^2"),colormap=:jet)
Colorbar(fig[2,2],hm3)
_,hm4 = heatmap(fig[2,3],xx,xx,angle.(ψdn),axis=(aspect=1,title=L"arg$\phi_{\downarrow}$"),colormap=:jet)
Colorbar(fig[2,4],hm4)
Label(fig[0,:],title)
fig

## --- plot spin distribution ---
xx = range(-1.1pi,1.1pi,64)
sp = cal_bloch_spin(Γ, ϕG, lat, xx, xx)
nsp= vec(sqrt.(sp[1].^2 .+ sp[2].^2))
f,_,hm = heatmap(xx,xx,sp[3],axis=(;aspect=1),figure=(size=(400,300),))

arrows!(xx, xx, sp[1], sp[2], arrowsize = nsp.*80, lengthscale = 2
    #, arrowcolor = nsp, linecolor = nsp, 
)
Colorbar(f[1,2],hm)
f
# sp_r1 = copy(sp)

## --- 2D bands ---
kmesh = mymesh([(lat.b[:,1].+lat.b[:,2])./(-2),lat.b[:,1],lat.b[:,2]],[64,64])
en = eigband(lat,kmesh,1:8)

f,_,_=surface(en[1,:,:],axis=(type=Axis3,),colormap=:bwr,colorrange=extrema(en[1:4,:,:]))
surface!(en[3,:,:],colormap=:bwr,colorrange=extrema(en[1:4,:,:]))
f

# save("data/en2d.hdf5",Dict("v0"=>lat.v0,"m0"=>lat.m0,"en"=>en))

## --- phase diagram ---
function phase_diagram(M0; Nopt=10,N1=10^5,N2=10^3)
    spin = Array{Float64}(undef,3,length(M0))
    Γ = [0.0,0.0]
    for i in eachindex(M0)
        println("\n M0:",M0[i])
        lat = set_lattice(4.0,M0[i], 0.25,0.2;mz=0.0)
        E0,ϕ0= eigenband(lat, Γ, 1:Nopt)
        init_gs = ComplexF64[1.0,cis(-0.25), zeros(Nopt-2)...] # in-plane phase

        ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=init_gs, Nstep=N1)
        mat = calmat(lat, Γ)
        ϕG,u0=imag_time_evl(mat, ϕG, lat; Nstep=N2)
        SpinHall.gaugephi0!(ϕG, ϕ0)
        spin[:,i].= real.(dot_spin(ϕG))
    end
    return spin
end
M1 = range(0.1,0.5,step=0.1)
sp1 = phase_diagram(M1; Nopt=16,N1=4*10^5,N2=2*10^3)
M2 = range(0.6,1.1,step=0.1)
sp2 = phase_diagram(M2; Nopt=16,N1=2*10^5,N2=2*10^3)
M3 = range(1.15,1.35,step=0.05)
sp3 = phase_diagram(M3; Nopt=16,N1=4*10^5,N2=2*10^3)
M4 = range(1.4,3.0,step=0.1)
sp4 = phase_diagram(M4; Nopt=16,N1=2*10^5,N2=2*10^3)
M0=[M1; M2; M3; M4]
sp_M = [sp1 sp2 sp3 sp4]
series(M0,[abs.(sp_M[3,:]) sqrt.(sp_M[1,:].^2 .+sp_M[2,:].^2)]',markersize=8,color=c)


save("data/phase_diagram.h5",
    Dict("M0"=>M0,"sp"=>sp_M,"xx"=>[xx;],
    "inplane_V4M1_x"=>sp_r2[1],
    "inplane_V4M1_y"=>sp_r2[2],
    "inplane_V4M1_z"=>sp_r2[3],
    "perp_V4M2_x"=>sp_r1[1],
    "perp_V4M2_y"=>sp_r1[2],
    "perp_V4M2_z"=>sp_r1[3],
    "g"=>[lat.g1,lat.g2])
)

## ---------------------------------------------------------------
#                    onsager relations
#  ---------------------------------------------------------------
η = 0.1
sz = Diagonal([fill(0.5,lat.NK); fill(-0.5,lat.NK)])
Jsz = [sz*ϕG; conj.(sz*ϕG)]
w = range(-2,2,161)

sc_xx = cal_SC_w(ϕG,w,lat,u0,Jsz; θ1=0.0,θ2=0.0,η=η)./lat.Sunit
cs_xx = cal_CS_w(ϕG,w,lat,u0,Jsz; θ1=0.0,θ2=0.0,η=η)./lat.Sunit
sc_xy = cal_SC_w(ϕG,w,lat,u0,Jsz; θ1=0.0,θ2=0.5pi,η=η)./lat.Sunit
cs_yx = cal_CS_w(ϕG,w,lat,u0,Jsz; θ1=0.5pi,θ2=0.0,η=η)./lat.Sunit


##
begin
    title = @sprintf("(v_0,m_0,g_{11},g_{12},\\eta,m_z)=(%.1f,%.1f,%.2f,%.4f,%.2f,%.2f)",lat.v0,lat.m0,lat.g1,lat.g2,η,lat.mz)|>latexstring
    f = Figure(size=(9,8).*80)

    ax1 = Axis(f[1,1],limits=(nothing,extrema(real.([sc_xx;cs_xx])).*1.1))
    scatterlines!(w,real.(sc_xx),label=L"\sigma^{sc}_{xx}",marker=:utriangle,markersize=6)
    scatterlines!(w,real.(cs_xx),label=L"\sigma^{cs}_{xx}",markersize=6)
    axislegend(position=:lt)

    ax2 = Axis(f[1,2],limits=(nothing,extrema(imag.([sc_xx;cs_xx])).*1.1))
    scatterlines!(w,imag.(sc_xx),label=L"\sigma^{sc}_{xx}",marker=:utriangle,markersize=6)
    scatterlines!(w,imag.(cs_xx),label=L"\sigma^{cs}_{xx}",markersize=6)
    axislegend(position=:lt)

    ax3 = Axis(f[2,1],limits=(nothing,extrema(real.([sc_xy;cs_yx])).*1.1),title="Re σ")
    scatterlines!(w,real.(sc_xy),label=L"\sigma^{sc}_{xy}",marker=:utriangle,markersize=6)
    scatterlines!(w,real.(cs_yx).*(-1),label=L"-\sigma^{cs}_{yx}",markersize=6)
    axislegend(position=:ct)

    ax4 = Axis(f[2,2],limits=(nothing,extrema(imag.([sc_xy;cs_yx])).*1.1),title="Im σ")
    scatterlines!(w,imag.(sc_xy),label=L"\sigma^{sc}_{xy}",marker=:utriangle,markersize=6)
    scatterlines!(w,imag.(cs_yx).*(-1),label=L"-\sigma^{cs}_{yx}",markersize=6)
    axislegend()

    Label(f[0,:],title)
    f
end


##
X = Dict("v0"=>lat.v0,"m0"=>lat.m0,"mz"=>lat.mz,"g1"=>lat.g1,"g2"=>lat.g2,"eta"=>η,"w"=>[w;])
X["sc_xx_re"] = real.(sc_xx)
X["sc_xx_im"] = imag.(sc_xx)

X["cs_xx_re"] = real.(cs_xx)
X["cs_xx_im"] = imag.(cs_xx)

X["sc_xy_re"] = real.(sc_xy)
X["sc_xy_im"] = imag.(sc_xy)

X["cs_yx_re"] = real.(cs_yx)
X["cs_yx_im"] = imag.(cs_yx)

save("data/bose_perpendicular.h5",X)
# save("data/bose_inplane.hdf5",X)












## ------------------ something for check -------------------------




## ---------------------------------------------------------------
#       (1) compare with central difference method
#  ---------------------------------------------------------------
q = [1e-4,-1e-4]
sc_q = cal_SC_qw(ϕG,q,[0.0],w,lat,u0,Jsz; θ2=0.0,η=η)./lat.Sunit # xx direction
dsc = (sc_q[:,1,1].-sc_q[:,2,1]).*(1im/(q[1]-q[2]))
cs_q = cal_CS_qw(ϕG,q,[0.0],w,lat,u0,Jsz; θ1=0.0,η=η)./lat.Sunit
dcs = (cs_q[:,1,1].-cs_q[:,2,1]).*(1im/(q[1]-q[2]))

begin
    title = @sprintf("(v_0,m_0,g_{11},g_{12},\\eta,m_z)=(%.1f,%.1f,%.2f,%.4f,%.2f,%.2f)",lat.v0,lat.m0,lat.g1,lat.g2,η,lat.mz)|>latexstring
    f = Figure(size=(9,8).*80)

    ax1 = Axis(f[1,1],limits=(nothing,extrema(real.(sc_xx)).*1.1))
    scatterlines!(w,real.(dsc),label=L"d\sigma^{sc}_{xx}",marker=:utriangle,markersize=6)
    scatterlines!(w,real.(sc_xx),label=L"\sigma^{sc}_{xx}",markersize=6)
    axislegend(position=:rt)

    ax2 = Axis(f[1,2],limits=(nothing,extrema(imag.(sc_xx)).*1.1))
    scatterlines!(w,imag.(dsc),label=L"d\sigma^{sc}_{xx}",marker=:utriangle,markersize=6)
    scatterlines!(w,imag.(sc_xx),label=L"\sigma^{sc}_{xx}",markersize=6)
    axislegend(position=:rt)

    ax3 = Axis(f[2,1],limits=(nothing,extrema(real.(cs_xx)).*1.1))
    scatterlines!(w,real.(dcs),label=L"d\sigma^{sc}_{xx}",marker=:utriangle,markersize=6)
    scatterlines!(w,real.(cs_xx),label=L"\sigma^{sc}_{xx}",markersize=6)
    axislegend(position=:rt)

    ax3 = Axis(f[2,2],limits=(nothing,extrema(imag.(cs_xx)).*1.1))
    scatterlines!(w,imag.(dcs),label=L"d\sigma^{sc}_{xx}",marker=:utriangle,markersize=6)
    scatterlines!(w,imag.(cs_xx),label=L"\sigma^{sc}_{xx}",markersize=6)
    axislegend(position=:rt)

    Label(f[0,:],title)
    f
end

## ---------------------------------------------------------------
#        (2)    check j_s = j_conv + j_τ
#  ---------------------------------------------------------------
H0 = calVmat(lat)
τ1 = (H0*sz .- sz*H0).*1im
Jτ = [τ1*ϕG; conj.(τ1*ϕG)]

sc1 = cal_SC1_w(ϕG,w,lat,u0; θ1=0.0,θ2=0.5pi,η=η)./lat.Sunit
sc2 = cal_SC2_w(ϕG,w,lat,u0,Jτ; θ1=0.0, θ2=0.5pi,η=η)./lat.Sunit
sc = cal_SC_w(ϕG,w,lat,u0,Jsz; θ1=0.0,θ2=0.5pi,η=η)./lat.Sunit

#   # xx direction
# sc1 = cal_SC1_w(ϕG,w,lat,u0; θ1=0.0,θ2=0.0,η=η)./lat.Sunit
# sc1.+= (dot_sz(ϕG)*1im./(w.+1im.*η))./lat.Sunit
# sc2 = cal_SC2_w(ϕG,w,lat,u0,Jτ; θ1=0.0, θ2=0.0,η=η)./lat.Sunit
# sc = cal_SC_w(ϕG,w,lat,u0,Jsz; θ1=0.0,θ2=0.0,η=η)./lat.Sunit

begin
    title = @sprintf("(v_0,m_0,g_{11},g_{12},\\eta)=(%.1f,%.1f,%.2f,%.4f,%.2f)",lat.v0,lat.m0,lat.g1,lat.g2,η)|>latexstring
    f = Figure(size=(9,4).*120)

    ax1 = Axis(f[1,1],limits=(nothing,(-0.1,0.1)),title=title)
    scatterlines!(w,real.(sc1),label=L"\sigma^{sc,1}_{yx}")
    scatterlines!(w,real.(sc2),label=L"\sigma^{sc,2}_{yx}")
    scatterlines!(w,real.(sc1.+sc2),label=L"\Sigma_i\sigma^{sc,i}_{yx}",markersize=10)
    scatterlines!(w,real.(sc),label=L"\sigma^{sc}_{xy}",marker=:utriangle,markersize=6)
    axislegend()

    ax2 = Axis(f[1,2],limits=(nothing,(-1,1).*0.15),title=title)
    scatterlines!(w,imag.(sc1),label=L"\sigma^{sc,1}_{yx}")
    scatterlines!(w,imag.(sc2),label=L"\sigma^{sc,2}_{yx}")
    scatterlines!(w,imag.(sc1.+sc2),label=L"\Sigma_i\sigma^{sc,i}_{yx}",markersize=10)
    scatterlines!(w,imag.(sc),label=L"\sigma^{sc}_{xy}",marker=:utriangle,markersize=6)
    axislegend()
    f
end



## 有效质量-无相互作用
Δ = 4e-5
Δk= [1.0,0.0].*Δ
k = [-Δk [0.0,0.0] Δk].+Γ
en = eigband(lat,k, 1:1)

m_e = 1/((en[1]+en[3]-2en[2])/Δ^2)
println("m/m^* = ", 0.5/m_e)


## 有效质量-平均场相互作用
en = Array{Float64}(undef,3)
for i in axes(k,2)
    Nopt = 10
    E0,ϕ0= eigenband(lat, k[:,i], 1:Nopt)
    init_gs = ComplexF64[0.0,1.0, zeros(Nopt-2)...]

    ϕG,u0,xopt=main_opt(E0, ϕ0, lat; gs=init_gs, Nstep=10^5)
    mat = calmat(lat, k[:,i])
    ϕG,u0,en[i]=imag_time_evl(mat, ϕG, lat; Nstep=2*10^3)
end
m_e = 1/((en[1]+en[3]-2en[2])/Δ^2)
println("m/m^* = ", 0.5/m_e)

