struct InverseWishart{P<:AbstractPDMat{T}, V<:AbstractVector{T}} where {T<:Real} <: CMD
    ψ::P
    v::V
    _nograd::Dict{Symbol,Bool}
end

function checknu(v::AbstractVector)
    length(v) == 1 || throw(DimensionMismatch("v must be a vector of length 1"))
    1
end

function InverseWishart(ψ::AbstractMatrix, v::AbstractVector)
    checknu(v)

    _nograd = Dict(
        :ψ => ψ isa NoGradArray,
        :v => v isa NoGradArray)
    ψ = _nograd[:ψ] ? ψ.data : ψ
    v = _nograd[:v] ? v.data : v

    InverseWishart(ψ, v, _nograd)
end

InverseWishart(ψ::AbstractVector, v::AbstractVector) = InverseWishart(DiagPDMat(ψ), v)
function InverseWishart(ψ::AbstractVector, dim::Int, v::AbstractVector)
    length(ψ) == 1 || throw(error(
        DimensionMismatch("ψ has to be a vector of length 1 to create a ScalPDMat")))
    InverseWishart(ScalPDMat(ψ[1],dim), v)
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

function lognormc0(p::InverseWishart)
    v = rate(p)
    n = length(p)
    (v*p/2)*log(2) + logmvgamma(p,v/2)
end

function _logpdf(p::InverseWishart, x::AbstractMatrix)
    ψ  = scale(p)
    v  = rate(p)
    n  = length(p)
    dψ = det(ψ)
    dx = det(x)

    v/2 * log(dψ) -(v+n+1)/2 * log(dx) - tr(ψ*inv(x))/2 - lognormc0(p)
end

logpdf(p::InverseWishart, x::AbstractMatrix) = _logpdf(p,x)
#logpdf(p::InverseWishart, x::AbstractVector) = error()

function Base.show(io::IO, p::InverseWishart)
    msg = "InverseWishart(ψ=$(summary(mean(p))), v=$(rate(p))"
    print(io, msg)
end
