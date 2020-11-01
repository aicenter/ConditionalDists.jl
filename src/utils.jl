struct SplitLayer{T<:Tuple}
    layers::T
end

SplitLayer(xs...) = SplitLayer(map(maybe_trainable, xs))

function (m::SplitLayer)(x)
    Tuple(layer(x) for layer in m.layers)
end

@functor SplitLayer


# for use as e.g. shared variance
struct TrainableVector{T<:AbstractArray}
    v::T
end
(v::TrainableVector)(x::AbstractVector) = v.v
(v::TrainableVector)(x::AbstractMatrix) = v.v .* reshape(fillsimilar(v.v,size(x,ndims(x)),1),1,:)
(v::TrainableVector)() = v.v
@functor TrainableVector

# for use as e.g. shared variance
struct TrainableScalar{T<:Real}
    s::AbstractVector{T}
end
TrainableScalar(x::Real) = TrainableScalar([x])
(s::TrainableScalar)(x::AbstractVector) = s.s[1]
(s::TrainableScalar)(x::AbstractMatrix) = fillsimilar(x,size(x,ndims(x)),1) .* s.s
@functor TrainableScalar

maybe_trainable(x) = x
maybe_trainable(x::AbstractArray) = TrainableVector(x)
maybe_trainable(x::Real) = TrainableScalar(x)

fillsimilar(x::AbstractArray, s::Tuple, value::Real) = fill!(similar(x, s...), value)
fillsimilar(x::AbstractArray, s, value::Real) = fill!(similar(x, s), value)
@non_differentiable fillsimilar(::Any, ::Any, ::Any)
