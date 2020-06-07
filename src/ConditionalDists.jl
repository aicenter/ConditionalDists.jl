module ConditionalDists

using Distributions
using LinearAlgebra
using Random
using StatsBase

# using DistributionsAD
#using PDMats

# import Distributions.mean,
#     Distributions.var,
#     Distributions.logpdf,
#     Base.rand,
#     Base.eltype,
#     Base.length

export mean,
    var,
    logpdf,
    condition

export TuringMvNormal
    #ConditionalMeanVarMvNormal,
    #BatchMvNormal


const CMD = ContinuousMultivariateDistribution
abstract type ConditionalDistribution end
abstract type ConditionalMvNormal <: ConditionalDistribution end

Base.length(p::ConditionalDistribution) = length(p.distribution)

include("mvnormal.jl")
# include("batch_mvnormal.jl")
# include("cond_mvnormal.jl")

end # module
