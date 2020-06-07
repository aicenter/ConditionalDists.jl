struct ConditionalMeanVarMvNormal{Td<:TuringMvNormal,Tm} <: ConditionalMvNormal
    distribution::Td
    mapping::Tm
end

const CMVDiagMvNormal = ConditionalMeanVarMvNormal{<:TuringDiagMvNormal}
const CMVScalMvNormal = ConditionalMeanVarMvNormal{<:TuringScalMvNormal}

function condition(p::CMVDiagMvNormal, z::AbstractVector)
    n = length(p)
    x = p.mapping(z)
    μ = x[1:n]
    σ = abs.(x[n+1:end])
    TuringMvNormal(μ,σ)
end

function condition(p::CMVDiagMvNormal, z::AbstractMatrix)
    n = length(p)
    x = p.mapping(z)
    μ = x[1:n,:]
    σ = abs.(x[n+1:end,:])
    BatchMvNormal(μ,σ)
end

function condition(p::CMVScalMvNormal, z::AbstractVector)
    n = length(p)
    x = p.mapping(z)
    μ = x[1:n]
    σ = abs(x[end])
    TuringMvNormal(μ,σ)
end

function condition(p::CMVScalMvNormal, z::AbstractMatrix)
    n = length(p)
    x = p.mapping(z)
    μ = x[1:n,:]
    σ = abs.(x[end,:])
    BatchMvNormal(μ,σ)
end

Distributions.mean(p::ConditionalMvNormal, z::AbstractVecOrMat) = mean(condition(p,z))
Distributions.var(p::ConditionalMvNormal, z::AbstractVecOrMat) = var(condition(p,z))
Distributions.rand(p::ConditionalMvNormal, z::AbstractVecOrMat) = rand(condition(p,z))
Distributions.logpdf(p::ConditionalMvNormal, x::AbstractVecOrMat, z::AbstractVecOrMat) =
    logpdf(condition(p,z), x)
