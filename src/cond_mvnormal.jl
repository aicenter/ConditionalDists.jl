"""
    ConditionalMvNormal(m)

Specialization of ConditionalDistribution for `MvNormal`s (for performance with
batches of inputs).  Does the same as ConditionalDistribution(MvNormal,m)
but for batches of inputs a `BatchMvNormal` is constructed that does
not just map over the batch but uses faster matrix multiplications.

```julia-repl
julia> m = SplitLayer(100,[100,100])
julia> p = ConditionalMvNormal(m)
julia> @time rand(p, rand(100,10000);
julia> @time rand(p, rand(100,10000);
julia> @time rand(p, rand(100,10000);
 0.047122 seconds (23 allocations: 38.148 MiB, 24.25% gc time)

julia> p = ConditionalDistribution(MvNormal, m)
julia> @time rand(p, rand(100,10000);
julia> @time rand(p, rand(100,10000);
julia> @time rand(p, rand(100,10000);
 3.626042 seconds (159.97 k allocations: 18.681 GiB, 34.92% gc time)
```

The mapping `m` must return a `Tuple` with mean and variance.
For a convenient way of doing this you can use a `SplitLayer`.


# Examples

`ConditionalMvNormal` and `SplitLayer` together support 3 different variance
configurations: fixed/unit variance, shared variance, and trained variance. The
three different configurations are explained below.

## Fixed/unit variance

Pass a function to the `SplitLayer` that returns the fixed variance with
appropriate batch dimensions
```julia-repl
julia> σ(x::Vector) = 2
julia> σ(x::Matrix) = ones(Float32,size(x,2)) .* 2
julia> m = SplitLayer(Dense(2,3), σ)
julia> p = ConditionalMvNormal(m)
julia> condition(p,rand(Float32,2)) isa DistributionsAD.TuringScalMvNormal
```
Passing a mapping with a single output array assumes unit variance.

## Shared variance

For a learned variance that is the same across the the whole batch, simply pass
a vector (or scalar) to the `SplitLayer`. The `SplitLayer` wraps vectors/scalars
into a `TrainableVector`s/`TrainableScalar`s.
```julia-repl
julia> m = SplitLayer(Dense(2,3), ones(Float32,3))
julia> p = ConditionalMvNormal(m)
julia> condition(p,rand(Float32,2)) isa DistributionsAD.TuringDiagMvNormal
```

## Trained variance

Simply pass another trainable mapping for the variance. By just supplying input
sizes to `SplitLayer` you can automatically create `Dense` layers with given
activation functions. In this example the second activation function makes sure
that the variance is always positive
```julia-repl
julia> m = SplitLayer(2,[3,1],[identity,abs])
julia> p = ConditionalMvNormal(m)
julia> condition(p,rand(Float32,2)) isa DistributionsAD.TuringScalMvNormal
```
"""
struct ConditionalMvNormal{Tm} <: AbstractConditionalDistribution
    mapping::Tm
end

function condition(p::ConditionalMvNormal, z::AbstractVector)
    (μ,σ) = mean_var(p.mapping(z))
    if length(σ) == 1
        σ = σ[1]
    end
    DistributionsAD.TuringMvNormal(μ,σ)  # for CuArrays/gradients
end

function condition(p::ConditionalMvNormal, z::AbstractMatrix)
    (μ,σ) = mean_var(p.mapping(z))
    if size(σ,1) == 1
        σ = dropdims(σ, dims=1)
    end
    BatchMvNormal(μ,σ)
end

# dispatches for different outputs from mappings
# general case
mean_var(x::Tuple) = x
# single output assumes σ=1
mean_var(x::AbstractVector) = (x, 1)
mean_var(x::AbstractMatrix) = (x, fillsimilar(x,size(x,2),1))
# fixed scalar variance
# mean_var(x::Tuple{<:AbstractVector,<:Real}) = x; is already coverged
function mean_var(x::Tuple{<:AbstractMatrix,<:Real})
    (μ,σ) = x
    (μ,fillsimilar(μ,size(μ,2),σ))
end

# TODO: this should be moved to DistributionsAD
Distributions.mean(p::TuringDiagMvNormal) = p.m
Distributions.mean(p::TuringScalMvNormal) = p.m
Distributions.var(p::TuringDiagMvNormal) = p.σ .^2
Distributions.var(p::TuringScalMvNormal) = p.σ^2

@functor ConditionalMvNormal
