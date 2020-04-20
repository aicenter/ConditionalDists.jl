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
    TuringDiagMvNormal(μ,σ)
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
    TuringScalMvNormal(μ,σ)
end

function condition(p::CMVScalMvNormal, z::AbstractMatrix)
    n = length(p)
    x = p.mapping(z)
    μ = x[1:n,:]
    σ = abs.(x[end,:])
    BatchMvNormal(μ,σ)
end

mean(p::ConditionalMvNormal, z::AbstractVector) = condition(p,z).m
var(p::ConditionalMvNormal, z::AbstractVector) = abs2.(condition(p,z).σ)
mean(p::ConditionalMvNormal, z::AbstractMatrix) = mean(condition(p,z))
var(p::ConditionalMvNormal, z::AbstractMatrix) = var(condition(p,z))
rand(p::ConditionalMvNormal, z::AbstractVecOrMat) = rand(condition(p,z))
logpdf(p::ConditionalMvNormal, x::AbstractVecOrMat, z::AbstractVecOrMat) =
    logpdf(condition(p,z), x)
