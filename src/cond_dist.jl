abstract type AbstractConditionalDistribution end
const ACD = AbstractConditionalDistribution

Distributions.mean(p::ACD, z::AbstractVecOrMat) = mean(condition(p,z))
Distributions.var(p::ACD, z::AbstractVecOrMat) = var(condition(p,z))
Distributions.rand(p::ACD, z::AbstractVecOrMat) = rand(condition(p,z))
Distributions.logpdf(p::ACD, x::AbstractVecOrMat, z::AbstractVecOrMat) = logpdf(condition(p,z), x)

"""
    ConditionalDistribution(T, m)

A conditional distribution that can map examples `x` to a distribution of type
`T` via the mapping `m` by calling `condition`.
There are no type restrictions on `T` so it could also be a function that
constructs and appropriate distribution (e.g. like TuringMvNormal).

# Examples
Moments, likelihoods, and samples can be obtained via e.g. `mean`, `logpdf`,
and `rand`:
```julia-repl
julia> using ConditionalDists, Flux
julia> m = ConditionalDists.SplitLayer(5,[2,2])
julia> p = ConditionalDistribution(MvNormal, m)
julia> rand(p, rand(5))
2-element Array{Float32,1}:
 0.2433714
 2.2054431
```
"""
struct ConditionalDistribution{Td,Tm} <: AbstractConditionalDistribution
    # TODO: would be nice if it was restricted to a `Distribution`
    DistConstructor::Td
    mapping::Tm
end

"""
    condition(p::ConditionalDistribution, x::AbstractVector)

Map an example `x` to the distribution type specified in `p`.
"""
function condition(p::ConditionalDistribution, x::AbstractVector)
    p.DistConstructor(p.mapping(x)...)
end

"""
    condition(p::ConditionalDistribution, x::AbstractMatrix)

Map a batch of examples `x` (examples as columns of `x`) to the distribution
type specified in `p`.
"""
function condition(p::ConditionalDistribution, xs::AbstractMatrix)
    ds = map(i -> condition(p, view(xs, :, i)), 1:size(xs,2))
    arraydist(ds)
end

function Base.show(io::IO, p::ConditionalDistribution)
    Np = nameof(typeof(p))
    Nd = nameof(p.DistConstructor)
    Nm = repr(p.mapping)
    print(io, "$Np($Nd, $Nm)")
end

Flux.@functor ConditionalDistribution
