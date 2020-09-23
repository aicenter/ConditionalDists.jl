struct SplitLayer
    layers::Tuple
end

function SplitLayer(in::Int, outs::Vector{Int}, act=identity)
    acts = [act for _ in 1:length(outs)]
    SplitLayer(in, outs, acts)
end

function (m::SplitLayer)(x)
    Tuple(layer(x) for layer in m.layers)
end

@functor SplitLayer

fillsimilar(x::AbstractArray, s::Tuple, value::Real) = fill!(similar(x, s...), value)
fillsimilar(x::AbstractArray, s, value::Real) = fill!(similar(x, s), value)
@non_differentiable fillsimilar(::Any, ::Any, ::Any)
