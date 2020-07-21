using Test
using LinearAlgebra
using Distributions
using DistributionsAD
using ConditionalDists
using Flux

using Zygote

using ConditionalDists: BatchMvNormal

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")
