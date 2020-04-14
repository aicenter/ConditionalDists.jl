module ConditionalDists

export ConditionalMeanVarMvNormal, ConditionalMeanMvNormal

using Distributions
using DistributionsAD

# The goal is to get conditional distributions that are defined by e.g. a
# MvNormal and a mapping f. They would be used like this:
#
# cd = ConditionalDistribution(f, d)
# mean(d,z)     -> compute mean/var/other modes
# rand(d,z)     -> sample
# logpdf(d,x,z) -> compute p(x|z)
#
# This can be achieved by defining a `condition` function that creates a proper
# distributions D form their conditional counter part CD:
#
#   condition(cd, z) = D(cd.f(z)...) -> dist
#
# The code below illustrates roughly how it could be done for a DiagMvNormals.
# The ConditionalMeanVarMvNormal implements the case were both mean and
# variance come from the mapping f. The ConditionalMeanMvNormal the case were
# only the mean comes from the mapping (i.e. its specific to each input) and the
# variance is shared for all inputs.

abstract type ConditionalDistribution end

Base.length(d::ConditionalDistribution) = length(d.d)
Distributions.rand(d::ConditionalDistribution, z::AbstractVector) = rand(condition(d, z))
Distributions.logpdf(d::ConditionalDistribution, x::AbstractVector, z::AbstractVector) =
    logpdf(condition(d, z), x)

const CMD = ContinuousMultivariateDistribution


struct ConditionalMeanVarMvNormal{Td<:CMD,Tf} <: ConditionalDistribution
    f::Tf
    d::Td  # TODO: this could be something like <: AbstractTuringMvNormal
end

# In this case d is in the only used to store information about length (and
# later for dispatch)
# Needs an additional constuctor for the ScalMvNormal case
function ConditionalMeanVarMvNormal(f::Tf, xlength::Int) where Tf
    d = TuringDiagMvNormal(zeros(xlength), zeros(xlength))
    ConditionalMeanVarMvNormal(f,d)
end

function condition(d::ConditionalMeanVarMvNormal{<:TuringDiagMvNormal}, z::AbstractVector)
    len = length(d)
    z = d.f(z)
    μ = z[1:len]
    σ = abs.(z[len+1:end]) .+ eps(eltype(z))
    TuringDiagMvNormal(μ,σ)
end

struct ConditionalMeanMvNormal{Td<:CMD,Tf} <: ConditionalDistribution
    f::Tf
    d::Td  # TODO: this could be something like <: TuringMvNormal
end

# also needs the construtor for the scalar case
function ConditionalMeanMvNormal(f::Tf, σ::AbstractVector) where Tf
    d = TuringDiagMvNormal(zeros(length(σ)), σ)
    ConditionalMeanMvNormal(f,d)
end

function condition(d::ConditionalMeanMvNormal{<:TuringDiagMvNormal}, z::AbstractVector)
    μ = d.f(z)
    TuringDiagMvNormal(μ,d.d.σ)
end

# TODO: these would have to be implemented for each Mean/MeanVar type
Distributions.mean(d::ConditionalDistribution, z::AbstractVector) = condition(d, z).m
# TODO: these would have to be implemented for each Diag/Scal type
Distributions.var(d::ConditionalDistribution, z::AbstractVector) = condition(d, z).σ

end # module
