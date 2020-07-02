using Revise

using Test
using LinearAlgebra
using PDMats
using Distributions
using ConditionalDists

using ReverseDiff, Tracker, ForwardDiff, Zygote, FiniteDifferences
using Flux


# include("mvnormal.jl")
include("cond_mvnormal.jl")
#include("batch_mvnormal.jl")
