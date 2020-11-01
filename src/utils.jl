"""
    SplitLayer(xs...)

A layer that calls a number of sublayers/mappings with the same input and
returns a tuple of their outputs. Can be used in a regular Flux model:

```julia-repl
julia> m = Chain(Dense(2,3), SplitLayer(Dense(3,2), x->x .* 2))
julia> length(params(m)) == 4
julia> (x,y) = m(rand(2))
(Float32[-1.0541434, 1.1694773], Float32[-3.1472511, -0.86115724, -0.39665926])
```
Comes with a convenient constructor for a SplitLayer built from Dense layers
with given activation(s):
```julia-repl
julia> m = Chain(Dense(2,3), SplitLayer(3, [2,5], σ))
julia> (x,y) = m(rand(2))
(Float32[0.3069554, 0.3362006], Float32[0.437131, 0.4982477, 0.6465078, 0.4523438, 0.5068563])
```

You can also provide just a vector / scalar that should be trained but have the
same value for all inputs (like a lonely bias vector). This functionality is
provided by the `TrainableVector`/`TrainableScalar` types. For vector inputs
they simply return the array they are wrapping. For matrix (i.e. batch) inputs
they return appropriately repeated arrays:
```julia-repl
julia> m = SplitLayer(Dense(2,3), ones(Float32,3))
julia> length(params(m)) == 3
julia> (x,y) = m(rand(2,5))
julia> size(y) == (3,5)
julia> y
3×3 Array{Float32,2}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
```
"""
struct SplitLayer{T<:Tuple}
    layers::T
end

SplitLayer(layers...) = SplitLayer(map(maybe_trainable, layers))

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
    TrainableScalar{T}(x::T) where T<:Real = new{T}([x])
end
TrainableScalar(x::T) where T<:Real = TrainableScalar{T}(x)
(s::TrainableScalar)(x::AbstractVector) = s.s[1]
(s::TrainableScalar)(x::AbstractMatrix) = fillsimilar(x,size(x,ndims(x)),1) .* s.s
@functor TrainableScalar

maybe_trainable(x) = x
maybe_trainable(x::AbstractArray) = TrainableVector(x)
maybe_trainable(x::Real) = TrainableScalar(x)

fillsimilar(x::AbstractArray, s::Tuple, value::Real) = fill!(similar(x, s...), value)
fillsimilar(x::AbstractArray, s, value::Real) = fill!(similar(x, s), value)
@non_differentiable fillsimilar(::Any, ::Any, ::Any)
