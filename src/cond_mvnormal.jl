"""
    ConditionalMvNormal(m)

Specialization of ConditionalDistribution for `MvNormal`s (for performance with
batches of inputs).  Does the same as ConditionalDistribution(MvNormal,m)
but for batches of inputs a `BatchMvNormal` is constructed that does
not just map over the batch but uses faster matrix multiplications.

The mapping `m` must return either a `Tuple` with mean and variance.
For a convenient way of do this you can use a `SplitLayer`.

or just a
mean vector. If the output of `m` is just a vector, the variance is assumed to
be a fixed unit variance.

# Examples
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
