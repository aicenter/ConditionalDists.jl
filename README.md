[![Build Status](https://travis-ci.com/aicenter/ConditionalDists.jl.svg?branch=master)](https://travis-ci.com/aicenter/ConditionalDists.jl)
[![codecov](https://codecov.io/gh/aicenter/ConditionalDists.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/aicenter/ConditionalDists.jl)

# ConditionalDists.jl

Conditional probability distributions powered by Flux.jl and Distributions.jl.

The conditional PDFs that are defined in this package can be used in
conjunction with Flux models to provide trainable mappings. As an example,
assume you want to learn the mapping from a conditional to an MvNormal

```julia
julia> using Distributions, DistributionsAD, Flux
julia> using ConditionalDists

julia> Flux.@functor ConditionalMvNormal  # make it a functor!

julia> xlength = 3
julia> zlength = 2
julia> batchsize = 10
julia> m = Dense(zlength, 2*xlength)
julia> d = MvNormal(zeros(xlength), ones(xlength))
julia> p = ConditionalMvNormal(d,m)

julia> # BatchDiagMvNormal
julia> res = condition(p, rand(zlength,batchsize))
julia> μ = mean(res)
julia> σ2 = var(res)
julia> @assert res isa ConditionalDists.BatchDiagMvNormal

julia> x = rand(xlength, batchsize)
julia> z = rand(zlength, batchsize)
julia> logpdf(p,x,z)
2×5 Array{Float32,2}:
 -4.75223  -8.37436   -6.79707  -2.32712   0.236871
 -6.60262   0.119544  -2.40393   7.17728  -9.87703 

julia> rand(cp, randn(xlen,10))  # sample from cpdf
```

The trainable parameters (W,b of the `Dense` layer and the shared variance of
`cpdf`) are accessible as usual through `Flux.params` (because we called `Flux.@functor`).
The next few lines show how to optimize `cp` to match a given Gaussian by
using the `kl_divergence` defined in
[IPMeasures.jl](https://github.com/aicenter/IPMeasures.jl).

...
