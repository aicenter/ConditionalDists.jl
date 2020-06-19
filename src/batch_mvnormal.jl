struct BatchMvNormal{Tm<:AbstractMatrix,Tσ<:AbstractVecOrMat}
    μ::Tm
    σ::Tσ
    BatchMvNormal{Tm,Ts}(μ,σ) where {Tm<:AbstractMatrix,Ts<:AbstractVecOrMat} = new(μ,σ)
end

BatchMvNormal(μ::AbstractMatrix{T},σ::AbstractVecOrMat{T}) where T<:Real =
    BatchMvNormal{typeof(μ),typeof(σ)}(μ,σ)

const BatchScalNormal = BatchMvNormal{<:AbstractVector}
const BatchDiagNormal = BatchMvNormal{<:AbstractMatrix}

BatchMvNormal(μ::AbstractMatrix, σ::AbstractVecOrMat) = BatchMvNormal(μ,sqrt.(σ))

Base.length(d::BatchMvNormal) = size(d.μ,1)
Base.eltype(d::BatchMvNormal) = eltype(d.μ)
Distributions.params(d::BatchMvNormal) = (d.μ, d.σ)
Distributions.mean(d::BatchMvNormal) = d.μ
Distributions.var(d::BatchDiagNormal) = d.σ .^2
Distributions.var(d::BatchScalNormal) = fill!(similar(d.σ,size(d.μ,1)),1) .* reshape(d.σ .^2,1,:)

function Distributions.rand(d::BatchMvNormal)
    μ = d.μ
    σ = d.σ
    r = randn!(similar(μ))
    μ .+ σ .* r
end

function Distributions.logpdf(d::BatchMvNormal, x::AbstractMatrix)
    T = eltype(d)
    n = length(d)
    μ = mean(d)
    σ2 = var(d)
    -(vec(sum(((x - μ).^2) ./ σ2 .+ log.(σ2), dims=1)) .+ n*log(T(2π))) / 2
end
