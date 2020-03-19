export Gaussian
export mean_var

"""
    Gaussian{T}

Gaussian defined with mean μ and variance σ2 that can be any `AbstractArray`.
The covariance is a diagonal matrix `Diagonal(σ2)`.

# Arguments
- `μ::AbstractArray`: mean of Gaussian
- `σ2::AbstractArray`: variance of Gaussian

# Example
```julia-repl
julia> using ConditionalDists

julia> p = Gaussian(zeros(3), ones(3))
Gaussian{Float64}(μ=3-element Array{Float64,1}, σ2=3-element Array{Float64,1})

julia> mean_var(p)
([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]))

julia> rand(p)
Tracked 3×1 Array{Float64,2}:
 -1.8102550562952886
  0.6218903591706907
 -0.8067583329396676
```
"""
struct Gaussian{T<:AbstractVector} <: ContinuousMultivariateDistribution
    μ::T
    σ::T
    _nograd::Dict{Symbol,Bool}
end

function Gaussian(μ::AbstractVector, σ::AbstractVector, d::Dict)
    Gaussian{typeof(μ)}(μ,σ,d)
end

function Gaussian(μ::AbstractVector, σ::AbstractVector)
    _nograd = Dict(
        :μ => μ isa NoGradArray,
        :σ => σ isa NoGradArray)
    μ = _nograd[:μ] ? μ.data : μ
    σ = _nograd[:σ] ? σ.data : σ
    Gaussian(μ, σ, _nograd)
end

Flux.@functor Gaussian

function Flux.trainable(p::Gaussian)
    ps = (;(k=>getfield(p,k) for k in keys(p._nograd) if !p._nograd[k])...)
end

length(p::Gaussian) = size(p.μ, 1)
eltype(p::Gaussian) = eltype(p.μ)
mean(p::Gaussian) = p.μ
var(p::Gaussian) = p.σ .* p.σ .+ eltype(p)(1e-8)
cov(p::Gaussian) = Diagonal(var(p))
mean_var(p::Gaussian) = (mean(p), var(p))

function rand(p::Gaussian, batchsize::Int=1)
    (μ, σ2) = mean_var(p)
    k = length(p)
    r = randn!(similar(μ, k, batchsize))
    μ .+ sqrt.(σ2) .* r
end

function _logpdf(p::Gaussian, x::AbstractArray{T}) where T
    @assert eltype(p) == T
    (μ, σ2) = mean_var(p)
    - (sum((x .- μ).^2 ./ σ2, dims=1) .+ sum(log.(σ2) .+ T(log(2π)))) ./ 2
end

# this is necessary to avoid method ambiguity with Distributions.jl ...
logpdf(p::Gaussian, x::AbstractVector) = _logpdf(p,x)
logpdf(p::Gaussian, X::AbstractMatrix) = _logpdf(p,X)

function Base.show(io::IO, p::Gaussian)
    msg = "Gaussian(μ=$(summary(mean(p))), σ2=$(summary(var(p))))"
    print(io, msg)
end
