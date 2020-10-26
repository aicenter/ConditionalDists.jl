struct SplitLayer{T<:Tuple}
    layers::T
end

SplitLayer(xs...) = SplitLayer(xs)

function (m::SplitLayer)(x)
    Tuple(layer(x) for layer in m.layers)
end

@functor SplitLayer

fillsimilar(x::AbstractArray, s::Tuple, value::Real) = fill!(similar(x, s...), value)
fillsimilar(x::AbstractArray, s, value::Real) = fill!(similar(x, s), value)
@non_differentiable fillsimilar(::Any, ::Any, ::Any)
