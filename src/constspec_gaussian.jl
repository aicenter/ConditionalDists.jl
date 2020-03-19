export const_mean, const_var, const_mean_var
export spec_mean, spec_var, spec_mean_var
export ConstSpecGaussian

struct ConstSpecGaussian{C<:CMD, S<:ACD} <: ACD
    cnst::C
    spec::S
end

ConstSpecGaussian(c::C, s::S) where {C,S} = ConstSpecGaussian{C,S}(c,s)

Flux.@functor ConstSpecGaussian

function eltype(p::ConstSpecGaussian)
    T = eltype(p.cnst)
    @assert T == eltype(p.spec)
    return T
end

function length(p::ConstSpecGaussian)
    l = length(p.cnst)
    @assert l == length(p.spec)
    return l
end

const_mean(p::ConstSpecGaussian) = mean(p.cnst)
const_var(p::ConstSpecGaussian) = var(p.cnst)
const_cov(p::ConstSpecGaussian) = Diagonal(const_var(p))
const_mean_var(p::ConstSpecGaussian) = mean_var(p.cnst)
const_rand(p::ConstSpecGaussian) = rand(p.cnst)

spec_mean(p::ConstSpecGaussian, z::AbstractArray) = mean(p.spec, z)
spec_var(p::ConstSpecGaussian, z::AbstractArray) = var(p.spec, z)
spec_cov(p::ConstSpecGaussian) = Diagonal(spec_var(p))
spec_mean_var(p::ConstSpecGaussian, z::AbstractArray) = mean_var(p.spec, z)
spec_rand(p::ConstSpecGaussian, z::AbstractArray) = rand(p.spec, z)

function mean(p::ConstSpecGaussian, z::AbstractArray)
    μc = repeat(const_mean(p), 1, size(z,2))
    μs = spec_mean(p, z)
    (μc, μs)
end

function var(p::ConstSpecGaussian, z::AbstractArray)
    σc = repeat(const_var(p), 1, size(z,2))
    σs = spec_var(p, z)
    (σc, σs)
end

mean_var(p::ConstSpecGaussian, z::AbstractArray) = (mean(p,z), var(p,z))

rand(p::ConstSpecGaussian, z::AbstractArray) = (const_rand(p), spec_rand(p,z))

function logpdf(p::ConstSpecGaussian, x::AbstractArray, z::AbstractArray)
    cllh = logpdf(p.cnst, x)
    sllh = logpdf(p.spec, x, z)
    cllh + sllh
end

function Base.show(io::IO, p::ConstSpecGaussian)
    msg = """$(typeof(p)):
     const = $(p.cnst)
     spec  = $(p.spec)
    """
    print(io, msg)
end
