module ConditionalDists

using LinearAlgebra
using Random
using StatsBase
using Distributions
using DistributionsAD

export condition

export ConditionalMeanVarMvNormal

const CMD = ContinuousMultivariateDistribution
abstract type ConditionalDistribution end
abstract type ConditionalMvNormal <: ConditionalDistribution end

Base.length(p::ConditionalDistribution) = length(p.distribution)

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")

end # module
