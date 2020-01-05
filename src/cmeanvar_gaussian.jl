export CMeanVarGaussian
export mean_var

"""
    CMeanVarGaussian{AbstractVar}(mapping)

Conditional Gaussian that maps an input z to a mean μx and a variance σ2x.
The mapping must output dimensions appropriate for the chosen variance type:
- DiagVar: μx = mapping(z)[1:end/2]; σ2 = mapping(z)[end/2+1:end]
- ScalarVar: μx = mapping(z)[1:end-1]; σ2 = mapping(z)[end:end]

# Arguments
- `mapping`: maps condition z to mean and variance (e.g. a Flux Chain)
- `AbstractVar`: one of the variance types: DiagVar, ScalarVar
- `T`: expected eltype. E.g. `rand` will try to sample arrays of this eltype.
  If the mapping returns a different eltype the output of `mean`,`variance`,
  and `rand` is not necessarily of eltype T.

# Example
```julia-repl
julia> p = CMeanVarGaussian{Float32,ScalarVar}(Dense(2, 3))
CMeanVarGaussian{Float32,ScalarVar}(mapping=Dense(2, 3))

julia> mean_var(p, ones(2))
(Float32[1.6191938; -0.437356], Float32[4.131034])

julia> rand(p, ones(2))
2×1 Array{Float32,2}:
 0.7168678
 0.16322285
```
"""
struct CMeanVarGaussian{V<:AbstractVar,M} <: AbstractCGaussian
    mapping::M
end

CMeanVarGaussian{V}(m::M) where {V,M} = CMeanVarGaussian{V,M}(m)

function mean_var(p::CMeanVarGaussian{DiagVar}, z::AbstractArray)
    ex = p.mapping(z)
    xlen = Int(size(ex, 1) / 2)
    μ = ex[1:xlen,:]
    σ = ex[xlen+1:end,:]
    T = eltype(ex)
    return μ, σ .* σ .+ T(1e-8)
end

function mean_var(p::CMeanVarGaussian{ScalarVar}, z::AbstractArray)
    ex = p.mapping(z)
    μ = ex[1:end-1,:]
    σ = ex[end:end,:]
    T = eltype(ex)
    return μ, σ .* σ .+ T(1e-8)
end

# make sure that parameteric constructor is called...
function Flux.functor(p::CMeanVarGaussian{V}) where V
    fs = fieldnames(typeof(p))
    nt = (; (name=>getfield(p, name) for name in fs)...)
    nt, y -> CMeanVarGaussian{V}(y...)
end

function Base.show(io::IO, p::CMeanVarGaussian{V}) where V
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    msg = "CMeanVarGaussian{$V}(mapping=$e)"
    print(io, msg)
end
