struct InverseWishart{P<:AbstractPDMat,V<:AbstractVector} <: CMD
    ν::V
    ψ::P
    _nograd::Dict{Symbol,Bool}
end

function checknu(ν::AbstractVector)
    length(ν) == 1 || throw(DimensionMismatch("ν must be a vector of length 1"))
    1
end

function InverseWishart(ν::AbstractVector, ψ::AbstractMatrix)
    checknu(ν)

    _nograd = Dict(
        :ψ => ψ isa NoGradArray,
        :ν => ν isa NoGradArray)
    ψ = _nograd[:ψ] ? ψ.data : ψ
    ν = _nograd[:ν] ? ν.data : ν

    InverseWishart(ν, ψ, _nograd)
end

InverseWishart(ν::AbstractVector, ψ::AbstractVector) = InverseWishart(ν, DiagPDMat(ψ))

function InverseWishart(ν::AbstractVector, ψ::AbstractMatrix, dim::Int)
    length(ψ) == 1 || throw(error(
        DimensionMismatch("ψ has to be a vector of length 1 to create a ScalPDMat")))
    InverseWishart(ν, ScalPDMat(ψ[1],dim))
end

Flux.@functor InverseWishart

function Flux.trainable(p::InverseWishart)
    ps = (;(k=>getfield(p,k) for k in keys(p._nograd) if !p._nograd[k])...)
end

length(p::InverseWishart) = size(p.ψ, 1)
eltype(p::InverseWishart) = eltype(p.ψ)
mean(p::InverseWishart) = scale(p) ./ (rate(p) - length(p) - 1)
scale(p::InverseWishart) = p.ψ
rate(p::InverseWishart) = p.ν[1]

rand(p::InverseWishart) = error()

function lognormc0(p::InverseWishart)
    ν = rate(p)
    n = length(p)
    (ν*p/2)*log(2) + logmvgamma(p,ν/2)
end

function _logpdf(p::InverseWishart, x::AbstractMatrix)
    ψ  = scale(p)
    ν  = rate(p)
    n  = length(p)
    dψ = det(ψ)
    dx = det(x)

    ν/2 * log(dψ) -(ν+n+1)/2 * log(dx) - tr(ψ*inv(x))/2 - lognormc0(p)
end

logpdf(p::InverseWishart, x::AbstractMatrix) = _logpdf(p,x)
#logpdf(p::InverseWishart, x::AbstractVector) = error()

function Base.show(io::IO, p::InverseWishart)
    msg = "InverseWishart(ψ=$(summary(mean(p))), ν=$(rate(p))"
    print(io, msg)
end
