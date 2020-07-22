module ConditionalDists

using LinearAlgebra
using Random
using StatsBase
using Distributions
using DistributionsAD

export condition

export ConditionalMvNormal

const CMD = ContinuousMultivariateDistribution
abstract type ConditionalDistribution end

Base.length(p::ConditionalDistribution) = length(p.distribution)

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")

end # module
