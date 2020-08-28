struct SplitLayer
    layers::Tuple
end

function SplitLayer(in::Int, outs::Vector{Int}, act=identity)
    acts = [act for _ in 1:length(outs)]
    SplitLayer(in, outs, acts)
end

function SplitLayer(in::Int, outs::Vector{Int}, acts::Vector)
    SplitLayer(Tuple(Dense(in,o,a) for (o,a) in zip(outs,acts)))
end

function (m::SplitLayer)(x)
    Tuple(layer(x) for layer in m.layers)
end

Flux.@functor SplitLayer


# TODO: replace with DistributionsAD.randnsimilar
randnsimilar(x::AbstractArray) = randn!(similar(x))

# nograd for randnsimilar
function ChainRulesCore.rrule(::typeof(randnsimilar), x)
    function randnsimilar_pullback(ΔQ)
        return (NO_FIELDS, Zero())
    end
    randnsimilar(x), randnsimilar_pullback
end

fillsimilar(x::AbstractArray, s::Tuple, value::Real) = fill!(similar(x, s...), value)
fillsimilar(x::AbstractArray, s, value::Real) = fill!(similar(x, s), value)

# nograd for fillsimilar
function ChainRulesCore.rrule(::typeof(fillsimilar), x, s, v)
    function fillsimilar_pullback(ΔQ)
        return (NO_FIELDS, Zero(), Zero(), Zero())
    end
    fillsimilar(x,s,v), fillsimilar_pullback
end
