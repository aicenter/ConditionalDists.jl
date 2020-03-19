[![Build Status](https://travis-ci.com/aicenter/ConditionalDists.jl.svg?branch=master)](https://travis-ci.com/aicenter/ConditionalDists.jl)
[![codecov](https://codecov.io/gh/aicenter/ConditionalDists.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/aicenter/ConditionalDists.jl)

# ConditionalDists.jl

Conditional probability distributions powered by Flux.jl and Distributions.jl.

The conditional PDFs that are defined in this package can be used in
conjunction with Flux models to provide trainable mappings. As an example,
consider a conditional Gaussian for which you want to learn a mapping and a
shared variance:

```julia
julia> using ConditionalDists;
julia> using Flux;

julia> xlen = 3; zlen = 2;
julia> T    = Float32;

julia> cpdf = CMeanGaussian{DiagVar}(Dense(xlen, zlen), ones(T,zlen)*10)
CMeanGaussian{DiagVar}(mapping=Dense(3, 2), σ2=2-element Array{Float32,1}

julia> X = randn(T, xlen, 10);
julia> Z = randn(T, zlen, 10);
julia> logpdf(cpdf, X, Z)  # compute p(X|Z)
1×10 Array{Float32,2}:
 -6.45847  -6.47164  -6.44917  -6.46053  -6.45961  -6.45457  -6.44526  -6.4592  -6.47359  -6.45476

julia> rand(cpdf, randn(T,xlen,10))  # sample from cpdf
2×5 Array{Float32,2}:
 -4.75223  -8.37436   -6.79707  -2.32712   0.236871
 -6.60262   0.119544  -2.40393   7.17728  -9.87703 
```

The trainable parameters (W,b of the Dense layer and the shared variance of
`cpdf`) are accesible as usual through `params`.  The next few lines show how
to optimize `cpdf` to match a given Gaussian by using the `kl_divergence` defined
in [IPMeasures.jl](https://github.com/aicenter/IPMeasures.jl).

```julia
julia> using IPMeasures;

julia> pdf = Gaussian(zeros(T,zlen), ones(T,zlen));
julia> loss(x) = sum(kl_divergence(cpdf, pdf, x));
julia> ps = params(cpdf);
julia> opt = ADAM(0.1);
julia> data = [(randn(T, xlen),) for i in 1:2000];
julia> Flux.train!(loss, ps, data, opt);
```

The learnt mean and variance are fairly close to a standard normal:
```julia
julia> mean_var(cpdf, randn(T,xlen))
(Float32[-0.03580121, 0.002174838], Float32[1.0000002; 1.0000002])

julia> rand(cpdf, rand(T,xlen,10))  # sample from trained cpdf
2×5 Array{Float32,2}:
 1.44779    0.437584   -0.047717   1.47545    0.436742
 0.596167  -0.0327809   0.327143  -0.591193  -2.62733
```
