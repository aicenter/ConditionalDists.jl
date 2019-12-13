export loglikelihood, rand
export AbstractVar, DiagVar, ScalarVar

abstract type AbstractCGaussian{T} <: AbstractCPDF{T} end

"""Abstract variance type"""
abstract type AbstractVar end

"""Diagonal variance represented as a vector"""
struct DiagVar <: AbstractVar end

"""Scalar variance represented as a one-element vector"""
struct ScalarVar <: AbstractVar end

function rand(p::AbstractCGaussian{T}, z::AbstractArray) where T
    (μ, σ2) = mean_var(p, z)
    r = randn!(similar(μ))
    μ .+ sqrt.(σ2) .* r 
end

function loglikelihood(p::AbstractCGaussian{T}, x::AbstractArray, z::AbstractArray) where T
    (μ, σ2) = mean_var(p, z)
    d = x - μ
    y = d .* d
    y = (1 ./ σ2) .* y .+ log.(σ2) .+ T(log(2π))
    # should we drop the extra dimensions here?
    # the max is there for 1D arrays
    -sum(y, dims=1:max(1, ndims(y)-1)) / 2
end

half_split(X::AbstractArray{T,1}) where T = X[1:Int(size(X,1)/2)], X[1+Int(size(X,1)/2):end]
half_split(X::AbstractArray{T,2}) where T = X[1:Int(size(X,1)/2),:], X[1+Int(size(X,1)/2):end,:]
half_split(X::AbstractArray{T,3}) where T = X[:,1:Int(size(X,2)/2),:], X[:,1+Int(size(X,3)/2):end,:]
half_split(X::AbstractArray{T,4}) where T = X[:,:,1:Int(size(X,3)/2),:], X[:,:,1+Int(size(X,3)/2):end,:]
half_split(X::AbstractArray) = error("splitting  only implemented for dim<=4")

last_split(X::AbstractArray{T,1}) where T = X[1:end-1], X[end:end]
last_split(X::AbstractArray{T,2}) where T = X[1:end-1,:], X[end:end,:]
last_split(X::AbstractArray{T,3}) where T = X[:,1:end-1,:], X[:,end:end,:]
last_split(X::AbstractArray{T,4}) where T = X[:,:,1:end-1,:], X[:,:,end:end,:]
last_split(X::AbstractArray) = error("splitting  only implemented for dim<=4")

# this is not backpropagable
# function half_split(X::AbstractArray)
#    # first get the dimension to be split
#     xlen = Int(size(X, ndims(X)-1)/2)
# 
#     # now get the axes
#     axs1 = axs2 = [collect(ax) for ax in axes(X)]
#     axs1[end-1] = 1:xlen
#     axs2[end-1] = xlen+1:xlen*2
# 
#     X[axs1...], X[axs2...]
# end

# function half_split2(X::AbstractArray)
    # first get the dimension to be split
    # nd = ndims(X)
#     nd = length(size(X))
#     xlen = Int(size(X, nd-1)/2)
# 
    # now get the axes
#     inds1 = vcat([Colon() for _ in 1:nd-2]..., [1:xlen], Colon())
#     inds2 = vcat([Colon() for _ in 1:nd-2]..., [xlen+1:xlen*2], Colon())
    
#     X[inds1...], X[inds2...]
# end
