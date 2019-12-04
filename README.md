[![Build Status](https://travis-ci.com/aicenter/ConditionalDists.jl.svg?branch=master)](https://travis-ci.com/aicenter/ConditionalDists.jl)
[![codecov](https://codecov.io/gh/aicenter/ConditionalDists.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/AIC-ML/ConditionalDists.jl)

# ConditionalDists.jl
Conditional probability distributions powered by Flux.jl.
The conditional PDFs that are defined in this package can be used
in conjunction with Flux models as mappings. As an example, consider
a conditional Gaussian for which you want to learn a mapping and a 
shared variance:

```julia
julia> p = CMeanGaussian{Float32,DiagVar}(Dense(2,3),ones(Float32,3))
CMeanGaussian{Float32}(mapping=Dense(2, 3), σ2=3-element Array{Float32,1}

julia> length(params(p)) == 3
true

julia> mean_var(p,ones(2))
(Float32[1.8698889, -0.24418116, 0.76614076], Float32[1.0, 1.0, 1.0])

julia> rand(p, ones(2))
3-element Array{Float32,2}:
 0.1829532154926673
 0.1235498922955946
 0.0767166501426535
```
