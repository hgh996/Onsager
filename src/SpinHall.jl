module SpinHall

using LinearAlgebra #,MKL
# BLAS.set_num_threads(1)

include("utilities.jl")
include("Bloch.jl")

using Trapz
using BlackBoxOptim
using JLD2,FileIO
include("GroundState.jl")
include("pwBdG.jl")

using ChunkSplitters
include("Hall.jl")

include("Symm.jl")

end # module SpinHall
