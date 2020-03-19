using Test
using Suppressor
using Random
using Flux
using ConditionalDists

if Flux.use_cuda[] 
	using CuArrays, GPUArrays
	GPUArrays.allowscalar(false)
end

include("gaussian.jl")
include("nogradarray.jl")
include("cmeanvar_gaussian.jl")
include("cmean_gaussian.jl")
include("constspec_gaussian.jl")
