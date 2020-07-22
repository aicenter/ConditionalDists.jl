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
Flux.@functor ConditionalMvNormal

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")
