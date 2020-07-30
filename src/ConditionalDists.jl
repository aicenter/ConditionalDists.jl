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

Base.length(p::ConditionalDistribution) = p.xlength

_randn_init(x) = randn!(similar(x))

# nograd for _randn_init
function rrule(::typeof(_randn_init), x)
    function _randn_init_pullback(ΔQ)
        return (ChainRulesCore.NOFIELDS, ChainRulesCore.Zero())
    end
    _randn_init(x), _randn_init_pullback
end

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")

end # module
