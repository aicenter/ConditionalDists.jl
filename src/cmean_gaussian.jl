export CMeanGaussian
export mean_var, variance

"""
    CMeanGaussian{AbstractVar}(mapping, σ2, xlength)

Conditional Gaussian that maps an input z to a mean μx. The variance σ2 is
shared for all datapoints.  The mapping must output dimensions appropriate for
the chosen variance type:

# Arguments
- `mapping`: maps condition z to μ=mapping(z) (e.g. a Flux Chain)
- `σ2`: shared variance for all datapoints
- `xlength`: length of mean/variance vectors. Only needed for `ScalarVar`
- `AbstractVar`: one of the variance types: DiagVar, ScalarVar

# Example
```julia-repl
julia> p = CMeanGaussian{DiagVar}(Dense(2,3),ones(Float32,3))
CMeanGaussian{DiagVar}(mapping=Dense(2, 3), σ2=3-element Array{Float32,1}

julia> mean_var(p,ones(2))
(Float32[1.8698889, -0.24418116, 0.76614076], Float32[1.0, 1.0, 1.0])

julia> rand(p, ones(2))
3-element Array{Float32,2}:
 0.1829532154926673
 0.1235498922955946
 0.0767166501426535
```
"""
struct CMeanGaussian{V<:AbstractVar,S<:AbstractArray,M} <: AbstractCGaussian
    mapping::M
    σ::S
    xlength::Int
    _nograd::Dict{Symbol,Bool}
end

function CMeanGaussian{V}(m::M, σ, xlength::Int) where {V,M}
    _nograd = Dict(:σ => σ isa NoGradArray)
    σ = _nograd[:σ] ? σ.data : σ
    S = typeof(σ)
    CMeanGaussian{V,S,M}(m, σ, xlength, _nograd)
end

CMeanGaussian{DiagVar}(m, σ) = CMeanGaussian{DiagVar}(m, σ, size(σ,1))

mean(p::CMeanGaussian, z::AbstractArray) = p.mapping(z)

function variance(p::CMeanGaussian{DiagVar}, z::AbstractArray)
    T = eltype(p.σ)
    σ2 = p.σ .* p.σ .+ T(1e-8)
    repeat(σ2, outer=(1,size(z,2)))
end

function variance(p::CMeanGaussian{ScalarVar}, z::AbstractArray)
    T = eltype(p.σ)
    σ2 = p.σ .* p.σ .* fill!(similar(p.σ, p.xlength), 1) .+ T(1e-8)
    repeat(σ2, outer=(1,size(z,2)))
end

mean_var(p::CMeanGaussian, z::AbstractArray) = (mean(p, z), variance(p, z))

# make sure that parameteric constructor is called...
function Flux.functor(p::CMeanGaussian{V,S,M}) where {V,S,M}
    fs = fieldnames(typeof(p))
    nt = (; (name=>getfield(p, name) for name in fs)...)
    nt, y -> CMeanGaussian{V,S,M}(y...)
end

function Flux.trainable(p::CMeanGaussian)
    ps = [getfield(p,k) for k in keys(p._nograd) if !p._nograd[k]]
    (p.mapping, ps...)
end

function Base.show(io::IO, p::CMeanGaussian{V}) where V
    e = repr(p.mapping)
    e = sizeof(e)>50 ? "($(e[1:47])...)" : e
    m = "CMeanGaussian{$V}(mapping=$e, σ2=$(summary(p.σ))"
    print(io, m)
end
