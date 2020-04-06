struct InverseGamma{A<:AbstractVector} <: CMD
    α::A
    β::A
    xlength::Int
    _nograd::Dict{Symbol,Bool}
end

function InverseGamma(α::AbstractVector, β::AbstractVector, xlength::Int)
    A, B = eltype(α), eltype(β)
    A == B || throw(ArgumentError("eltype of α and β must match"))

    length(α) == 1 || length(β) == 1 || throw(
        DimensionMismatch("α and β must by of length one"))

    _nograd = Dict(
        :α => α isa NoGradArray,
        :β => β isa NoGradArray)
    α = _nograd[:α] ? α.data : α
    β = _nograd[:β] ? β.data : β

    InverseGamma(α,β,xlength,_nograd)
end

InverseGamma(α::Real, β::Real, xlength::Int) = InverseGamma([α], [β], xlength)

function InverseGamma(α::Real, β::Real, xlength::Int, nograd::Bool)
    if nograd 
        InverseGamma(NoGradArray([α]), NoGradArray([β]), xlength)
    else
        InverseGamma(α,β,xlength)
    end
end

Flux.@functor InverseGamma

function Flux.trainable(p::InverseGamma)
    ps = (;(k=>getfield(p,k) for k in keys(p._nograd) if !p._nograd[k])...)
end

length(p::InverseGamma) = p.xlength
eltype(p::InverseGamma) = eltype(p.α)
shape(p::InverseGamma) = p.α[1]
rate(p::InverseGamma) = p.β[1]

function mean(p::InverseGamma)
    p.α[1] > 1 || throw(DomainError("mean is only defined for α>1"))
    rate(p) / (shape(p) - 1)
end

function var(p::InverseGamma)
    α,β = shape(p), rate(p)
    α > 2 || throw(error(DomainError("variance is only defined for α>2")))
    β^2 / ((α-1)^2 * (α-2))
end

mode(p::InverseGamma) = rate(p) / (shape(p) + 1)

function _logpdf(p::InverseGamma, x::AbstractArray)
    α,β = shape(p), rate(p)
    sum(α*log(β) - loggamma(α) .+ (-α-1)*log.(x) - β ./ x, dims=1)
    #sum((-α-1)*log.(x) - (β ./ x), dims=1)
end

logpdf(p::InverseGamma, x::AbstractVector) = _logpdf(p,x)
logpdf(p::InverseGamma, x::AbstractMatrix) = _logpdf(p,x)

function Base.show(io::IO, p::InverseGamma)
    msg = "InverseGamma(α=$(shape(p)), β=$(rate(p)))"
    print(io,msg)
end
