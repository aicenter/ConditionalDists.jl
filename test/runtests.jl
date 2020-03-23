using Test
using Random
using Flux
using ConditionalDists
using LinearAlgebra

if Flux.use_cuda[] 
	using CuArrays, GPUArrays
	GPUArrays.allowscalar(false)
end

include("gaussian.jl")
include("nogradarray.jl")
include("cmeanvar_gaussian.jl")
include("cmean_gaussian.jl")
include("constspec_gaussian.jl")
