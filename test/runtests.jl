using Revise

using Test
using LinearAlgebra
using Distributions
using ConditionalDists

using Flux
using ReverseDiff, Tracker, ForwardDiff, Zygote


include("mvnormal.jl")
include("cond_mvnormal.jl")
include("batch_mvnormal.jl")
