module ConditionalDists

using Distributions
using LinearAlgebra
using Random
using StatsBase

export mean, var, logpdf, condition

export TuringMvNormal,
    BatchMvNormal,
    ConditionalMeanVarMvNormal

const CMD = ContinuousMultivariateDistribution
abstract type ConditionalDistribution end
abstract type ConditionalMvNormal <: ConditionalDistribution end

Base.length(p::ConditionalDistribution) = length(p.distribution)

include("mvnormal.jl")
include("batch_mvnormal.jl")
include("cond_mvnormal.jl")

end # module
