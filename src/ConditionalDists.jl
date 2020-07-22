module ConditionalDists

using LinearAlgebra
using Random
using StatsBase
using Distributions
using DistributionsAD

using Zygote # for @nograd calls...
# TODO: can we instead use:
# using ZygoteRules
# ignore(f) = f()
# ZygoteRules.@adjoint ignore(f) = f(), _ -> nothing
# or: https://github.com/JuliaDiff/ChainRulesCore.jl/issues/150

export condition

export ConditionalMvNormal

const CMD = ContinuousMultivariateDistribution
abstract type ConditionalDistribution end

Base.length(p::ConditionalDistribution) = length(p.distribution)

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")

end # module
