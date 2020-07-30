module ConditionalDists

using LinearAlgebra
using Random
using StatsBase
using Distributions
using DistributionsAD

export condition

export ConditionalDistribution
export ConditionalMvNormal

abstract type AbstractConditionalDistribution end
const ACD = AbstractConditionalDistribution

Distributions.mean(p::ACD, z::AbstractVecOrMat) = mean(condition(p,z))
Distributions.var(p::ACD, z::AbstractVecOrMat) = var(condition(p,z))
Distributions.rand(p::ACD, z::AbstractVecOrMat) = rand(condition(p,z))
Distributions.logpdf(p::ACD, x::AbstractVecOrMat, z::AbstractVecOrMat) = logpdf(condition(p,z), x)

struct ConditionalDistribution{Td,Tm} <: AbstractConditionalDistribution
    distribution::Td  # TODO: would be nice if this was restricted to Distribution
    mapping::Tm
end

function condition(p::ConditionalDistribution, x::AbstractVector)
    p.distribution(p.mapping(x)...)
end

function condition(p::ConditionalDistribution, xs::AbstractMatrix)
    ds = map(i -> condition(p, xs[:,i]), 1:size(xs,2))
    arraydist(ds)
end

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")

end # module
