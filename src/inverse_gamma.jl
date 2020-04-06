struct InverseWishart{P<:AbstractMatrix, V<:AbstractVector} <: CMD
    ψ::P
    v::V
    _nograd::Dict{Symbol,Bool}
end

function checkeltype(ψ::AbstractArray, v::AbstractArray)
    T = eltype(ψ)
    V = eltype(v)
    T == V || throw(ArgumentError("Both arrays must have same eltype"))
    return T
end

function checknu(v::AbstractVector)
    length(v) == 1 || throw(DimensionMismatch("v must be a vector of length 1"))
    1
end

function InverseWishart(ψ::AbstractMatrix, v::AbstractVector)
    LinearAlgebra.checksquare(ψ)
    LinearAlgebra.checkpositivedefinite(ψ)
    checknu(v)
    checkeltype(ψ,v)

    _nograd = Dict(
        :ψ => ψ isa NoGradArray,
        :v => v isa NoGradArray)
    ψ = _nograd[:ψ] ? ψ.data : ψ
    v = _nograd[:v] ? v.data : v

    InverseWishart(ψ, v, _nograd)
end

Flux.@functor InverseWishart

function Flux.trainable(p::InverseWishart)
    ps = (;(k=>getfield(p,k) for k in keys(p._nograd) if !p._nograd[k])...)
end

length(p::InverseWishart) = size(p.ψ, 1)
eltype(p::InverseWishart) = eltype(p.ψ)
mean(p::InverseWishart) = scale(p) ./ (rate(v) - length(p) - 1)
scale(p::InverseWishart) = p.ψ
rate(p::InverseWishart) = p.v[1]

rand(p::InverseWishart) = error()

function normc0(p::InverseWishart)
    v = rate(p)
    n = length(p)
    2^(v*p/2) * logmvgamma(p,v/2)
end

function _logpdf(p::InverseWishart, x::AbstractMatrix)
    ψ  = scale(p)
    v  = rate(p)
    n  = length(p)
    dψ = det(ψ)
    dx = det(x)

    v/2 * log(dψ) -(v+n+1)/2 * log(dx) - tr(ψ*inv(x))/2 - log(normc0(p))
end

logpdf(p::InverseWishart, x::AbstractMatrix) = _logpdf(p,x)
#logpdf(p::InverseWishart, x::AbstractVector) = error()

function Base.show(io::IO, p::InverseWishart)
    msg = "InverseWishart(ψ=$(summary(mean(p))), v=$(rate(p))"
    print(io, msg)
end
