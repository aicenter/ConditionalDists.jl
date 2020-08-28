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
