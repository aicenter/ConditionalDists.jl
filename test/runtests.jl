using Test
using LinearAlgebra
using PDMats
using Distributions
using DistributionsAD
using ConditionalDists
using Flux

using DistributionsAD: TuringMvNormal
using ConditionalDists: BatchMvNormal

Flux.@functor ConditionalDists.BatchDiagMvNormal
Flux.@functor ConditionalDists.BatchScalMvNormal
Flux.@functor DistributionsAD.TuringDiagMvNormal
Flux.@functor DistributionsAD.TuringScalMvNormal
Flux.@functor ConditionalDistribution
Flux.@functor ConditionalMvNormal

struct SplitLayer
    layers::Tuple
end
function SplitLayer(input::Int, outputs::Array{Int,1}, act=abs)
    SplitLayer(Tuple(Dense(input,out,act) for out in outputs))
end
function (m::SplitLayer)(x)
    Tuple(layer(x) for layer in m.layers)
end
Flux.@functor SplitLayer

include("cond_dist.jl")
include("batch_mvnormal.jl")
include("cond_mvnormal.jl")
