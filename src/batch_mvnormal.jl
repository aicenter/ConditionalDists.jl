struct BatchScalMvNormal{Tm<:AbstractMatrix,Tσ<:AbstractVector} <: ContinuousMatrixDistribution
    μ::Tm
    σ::Tσ
end

struct BatchDiagMvNormal{Tm<:AbstractMatrix,Tσ<:AbstractMatrix} <: ContinuousMatrixDistribution
    μ::Tm
    σ::Tσ
end

const BMN = Union{BatchScalMvNormal,BatchDiagMvNormal}

BatchMvNormal(μ::AbstractMatrix{T}, σ::AbstractVector{T}) where T<:Real = BatchScalMvNormal(μ,σ)
BatchMvNormal(μ::AbstractMatrix{T}, σ::AbstractMatrix{T}) where T<:Real = BatchDiagMvNormal(μ,σ)

#Base.length(d::BatchMvNormal) = size(d.μ,1)
Base.eltype(d::BMN) = eltype(d.μ)
Distributions.params(d::BMN) = (d.μ, d.σ)
Distributions.mean(d::BMN) = d.μ
Distributions.var(d::BatchDiagMvNormal) = d.σ .^2
Distributions.var(d::BatchScalMvNormal) = fillsimilar(d.σ,size(d.μ,1),1) .* reshape(d.σ .^2,1,:)

function Distributions.rand(d::BatchDiagMvNormal)
    μ, σ = d.μ, d.σ
    r = DistributionsAD.adapt_randn(Random.GLOBAL_RNG, μ, size(μ)...)
    μ .+ σ .* r
end

function Distributions.rand(d::BatchScalMvNormal)
    μ, σ = d.μ, reshape(d.σ, 1, :)
    r = DistributionsAD.adapt_randn(Random.GLOBAL_RNG, μ, size(μ)...)
    μ .+ σ .* r
end

function Distributions.logpdf(d::BMN, x::AbstractMatrix{T}) where T<:Real
    n = size(d.μ,1)
    μ = mean(d)
    σ2 = var(d)
    -(vec(sum(((x - μ).^2) ./ σ2 .+ log.(σ2), dims=1)) .+ n*log(T(2π))) ./ 2
end
