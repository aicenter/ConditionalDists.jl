const AbstractVar{T} = Union{<:Cholesky{<:T},<:AbstractVector{<:T},T} where T<:Real

struct TuringMvNormal{Tm<:AbstractVector, Ts<:AbstractVar} <: ContinuousMultivariateDistribution
    μ::Tm
    σ::Ts
    TuringMvNormal{Tm,Ts}(μ,σ) where {Tm<:AbstractVector,Ts<:AbstractVar} = new(μ,σ)
end

# TODO: not sure if the constructors are good. I am trying to prevent mixed Float32/Float64 types... #
TuringMvNormal(μ::AbstractVector{T}, σ::AbstractVar{T}) where T<:Real =
    TuringMvNormal{typeof(μ),typeof(σ)}(μ,σ)

const TuringDenseMvNormal = TuringMvNormal{<:AbstractVector,<:Cholesky}
const TuringDiagMvNormal = TuringMvNormal{<:AbstractVector,<:AbstractVector}
const TuringScalMvNormal = TuringMvNormal{<:AbstractVector,<:Real}

# Constructors
TuringMvNormal(μ::AbstractVector, Σ::AbstractMatrix) = TuringMvNormal(μ, cholesky(Σ))
TuringMvNormal(μ::AbstractVector, D::Diagonal) = TuringMvNormal(m, sqrt.(D.diag))
TuringMvNormal(μ::AbstractVector{T}, A::UniformScaling{T}) where T = TuringMvNormal(μ, sqrt(A.λ))

TuringMvNormal(d::Int, σ::Real) = TuringMvNormal(zeros(typeof(σ),d), σ)
TuringMvNormal(σ::AbstractVector) = TuringMvNormal(fill!(similar(σ,length(σ)),0), σ)
TuringMvNormal(Σ::AbstractMatrix) = TuringMvNormal(fill!(similar(Σ, size(Σ,1)),0), Σ)


# Distributions.jl API

Base.eltype(d::TuringMvNormal) = eltype(d.μ)
Base.length(d::TuringMvNormal) = length(d.μ)
Base.size(d::TuringMvNormal) = (length(d),)
Distributions.params(d::TuringMvNormal) = (d.μ, d.σ)
Distributions.mean(d::TuringMvNormal) = d.μ

Distributions.var(d::TuringDenseMvNormal) = diag(Matrix(d.σ))
Distributions.var(d::TuringDiagMvNormal) = d.σ .^2
Distributions.var(d::TuringScalMvNormal) = fill!(similar(d.μ,length(d)), d.σ^2)
Distributions.cov(d::TuringDenseMvNormal) = d.σ
Distributions.cov(d::TuringDiagMvNormal) = Diagonal(d.σ)
Distributions.cov(d::TuringScalMvNormal) = I * d.σ^2

Distributions.rand(d::TuringMvNormal, n::Int...) = rand(Random.GLOBAL_RNG, d, n...)

function Distributions.rand(rng::Random.AbstractRNG, d::TuringDenseMvNormal, n::Int...)
    return d.μ .+ d.σ.U' * randn(rng, length(d), n...)
end

function Distributions.rand(rng::Random.AbstractRNG, d::TuringDiagMvNormal, n::Int...)
    return d.μ .+ d.σ .* randn(rng, length(d), n...)
end

function Distributions.rand(rng::Random.AbstractRNG, d::TuringScalMvNormal, n::Int...)
    return d.μ .+ d.σ .* randn(rng, length(d), n...)
end

function _logpdf(d::TuringScalMvNormal, x::AbstractVector)
    T = eltype(d)
    return -(length(x) * log(T(2π) * abs2(d.σ)) + sum(abs2.((x .- d.μ) ./ d.σ))) / 2
end
function _logpdf(d::TuringScalMvNormal, x::AbstractMatrix)
    T = eltype(d)
    return -(size(x, 1) * log(T(2π) * abs2(d.σ)) .+ vec(sum(abs2.((x .- d.μ) ./ d.σ), dims=1))) ./ 2
end

function _logpdf(d::TuringDiagMvNormal, x::AbstractVector)
    T = eltype(d)
    return -(length(x) * log(T(2π)) + 2 * sum(log.(d.σ)) + sum(abs2.((x .- d.μ) ./ d.σ))) / 2
end
function _logpdf(d::TuringDiagMvNormal, x::AbstractMatrix)
    T = eltype(d)
    return -((size(x, 1) * log(T(2π)) + 2 * sum(log.(d.σ))) .+ vec(sum(abs2.((x .- d.μ) ./ d.σ), dims=1))) ./ 2
end

function _logpdf(d::TuringDenseMvNormal, x::AbstractVector)
    T = eltype(d)
    return -(length(x) * log(T(2π)) + logdet(d.σ) + sum(abs2.(d.σ.U' \ (x .- d.μ)))) / 2
end
function _logpdf(d::TuringDenseMvNormal, x::AbstractMatrix)
    T = eltype(d)
    return -((size(x, 1) * log(T(2π)) + logdet(d.σ)) .+ vec(sum(abs2.(d.σ.U' \ (x .- d.μ)), dims=1))) ./ 2
end

for T in (:AbstractVector, :AbstractMatrix)
    @eval Distributions.logpdf(d::TuringScalMvNormal, x::$T) = _logpdf(d, x)
    @eval Distributions.logpdf(d::TuringDiagMvNormal, x::$T) = _logpdf(d, x)
    @eval Distributions.logpdf(d::TuringDenseMvNormal, x::$T) = _logpdf(d, x)
end

function StatsBase.entropy(d::TuringDiagMvNormal)
    T = eltype(d.σ)
    return (length(d) * (T(log2π) + one(T)) / 2 + sum(log.(d.σ)))
end
