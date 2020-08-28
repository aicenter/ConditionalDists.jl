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
include("batch_mvnormal.jl")
include("cond_mvnormal.jl")
