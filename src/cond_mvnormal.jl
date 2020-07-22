struct ConditionalMvNormal{Td<:Distributions.AbstractMvNormal,Tm} <: ConditionalDistribution
    distribution::Td
    mapping::Tm
end

function condition(p::ConditionalMvNormal, z::AbstractVector)
    n = length(p)
    x = p.mapping(z)
    μ = x[1:n]
    σ = abs.(x[n+1:end])
    if length(σ) == 1
        σ = σ[1]
    end
    MvNormal(μ,σ)
end

function condition(p::ConditionalMvNormal, z::AbstractMatrix)
    n = length(p)
    x = p.mapping(z)
    μ = x[1:n,:]
    σ = abs.(x[n+1:end,:])
    if size(σ,1) == 1
        σ = dropdims(σ, dims=1)
    end
    BatchMvNormal(μ,σ)
end

Distributions.mean(p::ConditionalMvNormal, z::AbstractVecOrMat) = mean(condition(p,z))
Distributions.var(p::ConditionalMvNormal, z::AbstractVecOrMat) = var(condition(p,z))
Distributions.rand(p::ConditionalMvNormal, z::AbstractVecOrMat) = rand(condition(p,z))
Distributions.logpdf(p::ConditionalMvNormal, x::AbstractVecOrMat, z::AbstractVecOrMat) =
    logpdf(condition(p,z), x)
