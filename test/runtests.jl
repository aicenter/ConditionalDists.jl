using Test
using LinearAlgebra
using Distributions
using DistributionsAD
using ConditionalDists
using Flux

using ConditionalDists: BatchMvNormal

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")
