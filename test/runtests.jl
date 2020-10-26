using Test
using LinearAlgebra
using PDMats
using Distributions
using DistributionsAD
using ConditionalDists
using Flux

using DistributionsAD: TuringMvNormal
using ConditionalDists: BatchMvNormal, SplitLayer

include("cond_dist.jl")
include("cond_mvnormal.jl")
include("utils.jl")

# for the BatchMvNormal tests to work BatchMvNormals have to be functors!
include("batch_mvnormal.jl")
