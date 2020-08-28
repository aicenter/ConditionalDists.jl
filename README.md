[![Build Status](https://travis-ci.com/aicenter/ConditionalDists.jl.svg?branch=master)](https://travis-ci.com/aicenter/ConditionalDists.jl)
[![codecov](https://codecov.io/gh/aicenter/ConditionalDists.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/aicenter/ConditionalDists.jl)

# ConditionalDists.jl

Conditional probability distributions powered by Flux.jl and Distributions.jl.

The conditional PDFs that are defined in this package can be used in
conjunction with Flux models to provide trainable mappings. As an example,
assume you want to learn the mapping from a conditional to an MvNormal.  The
mapping `m` takes a vector `x` and maps it to a mean `μ` and a variance `σ`,
which can be achieved by using a `ConditionalDists.SplitLayer` as the last
layer in your network like the one below: The `SplitLayer` is constructed from
`N` `Dense` layers (with same input size) and outputs `N` vectors:
```julia
julia> m = SplitLayer(2, [3,4])
julia> m(rand(2))
(Float32[0.07946974, 0.13797458, 0.03939067], Float32[0.7006321, 0.37641272, 0.3586885, 0.82230335])
```

With the mapping `m` we can create a conditional distribution with trainable
mapping parameters:
```julia
julia> using ConditionalDists, Distributions
julia> using ConditionalDists: SplitLayer

julia> xlength = 3
julia> zlength = 2
julia> batchsize = 10
julia> m = SplitLayer(zlength, [xlength,xlength])
julia> p = ConditionalMvNormal(m)

julia> res = condition(p, rand(zlength))  # this also works for batches!
julia> μ = mean(res)
julia> σ2 = var(res)
julia> @assert res isa DistributionsAD.TuringDiagMvNormal

julia> x = rand(xlength, batchsize)
julia> z = rand(zlength, batchsize)
julia> logpdf(p,x,z)
julia> rand(p, randn(zlength, 10))
```
The trainable parameters (W,b of the `SplitLayer`) are accessible as usual
through `Flux.params`.  The next few lines show how to optimize `p` to match a
given Gaussian by using the `kl_divergence` defined in
[IPMeasures.jl](https://github.com/aicenter/IPMeasures.jl).
