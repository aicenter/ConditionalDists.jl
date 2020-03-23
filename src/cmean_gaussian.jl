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
struct CMeanGaussian{V<:AbstractVar,M,S<:AbstractVector} <: AbstractConditionalGaussian
    mapping::M
    σ::S
    xlength::Int
    _nograd::Dict{Symbol,Bool}
end

const CMGaussian = CMeanGaussian

CMeanGaussian{V}(m::M, σ::S, xlength::Int, d::Dict{Symbol,Bool}) where {V,M,S} =
    CMeanGaussian{V,M,S}(m,σ,xlength,d)

function CMeanGaussian{V}(m::M, σ::AbstractVector, xlength::Int) where {V,M}
    _nograd = Dict(:σ => σ isa NoGradArray)
    σ = _nograd[:σ] ? σ.data : σ
    CMeanGaussian{V}(m, σ, xlength, _nograd)
end

CMeanGaussian{ScalarVar}(m::M, σ::Number, xlength::Int) where M =
    CMeanGaussian{ScalarVar}(m, [σ], xlength)

CMeanGaussian{DiagVar}(m, σ) = CMeanGaussian{DiagVar}(m, σ, size(σ,1))

mean(p::CMeanGaussian, z::AbstractArray) = p.mapping(z)

_var(p::CMeanGaussian, z::AbstractArray) = p.σ .* p.σ .+ eltype(p)(1e-8)

function var(p::CMeanGaussian{DiagVar}, z::AbstractArray)
    σ2 = _var(p,z)
    σ2 * fill!(similar(σ2, 1, size(z,2)), 1)
end

function var(p::CMeanGaussian{ScalarVar}, z::AbstractArray)
    σ2 = _var(p,z)
    σ2 .* fill!(similar(p.σ, p.xlength, size(z,2)), 1)
end

function svar(p::CMeanGaussian{ScalarVar}, z::AbstractArray)
    σ2 = _var(p,z)
    σ2 * fill!(similar(p.σ, 1, size(z,2)), 1)
end

cov(p::CMeanGaussian{DiagVar}, z::AbstractArray) = map(Diagonal, eachcol(var(p,z)))
cov(p::CMeanGaussian{ScalarVar}, z::AbstractArray) = map(s->I*s, vec(svar(p,z)))
mean_var(p::CMeanGaussian, z::AbstractArray) = (mean(p,z), var(p,z))
mean_cov(p::CMeanGaussian, z::AbstractArray) = (mean(p,z), cov(p,z))
length(p::CMeanGaussian) = p.xlength
eltype(p::CMeanGaussian) = eltype(p.σ)


# make sure that parameteric constructor is called...
function Flux.functor(p::CMeanGaussian{V,M,S}) where {V,M,S}
    fs = fieldnames(typeof(p))
    nt = (; (name=>getfield(p, name) for name in fs)...)
    nt, y -> CMeanGaussian{V}(y...)
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
