module ConditionalDists

using LinearAlgebra
using Distributions
using DistributionsAD
using PDMats

import Distributions.mean,
    Distributions.var,
    Distributions.logpdf,
    Base.rand,
    Base.eltype,
    Base.length

export mean,
    var,
    logpdf,
    condition

export ConditionalMeanVarMvNormal,
    BatchMvNormal


const CMD = ContinuousMultivariateDistribution
const TuringMvNormal = Union{TuringScalMvNormal,TuringDiagMvNormal}
abstract type ConditionalDistribution end
abstract type ConditionalMvNormal <: ConditionalDistribution end

Base.length(p::ConditionalDistribution) = length(p.distribution)


include("batch_mvnormal.jl")
include("cond_mvnormal.jl")

end # module
