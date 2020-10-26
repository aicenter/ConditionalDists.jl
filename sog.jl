using Distributions
using DistributionsAD
using ConditionalDists
using Zygote


function _l(x::Matrix{T}, n, μ, σ2) where {T}
	-(vec(sum(((x - μ).^2) ./ σ2 .+ log.(σ2), dims=1)) .+ n*log(T(2π))) / 2
end

function _∇l(Δ, x, n, μ, σ2)
	Δ = Δ'
	δ = Δ .* (x - μ) ./ σ2
    (- δ, nothing, δ, Δ .* (((x - μ).^2 ./ (σ2.^2))  - 1 ./ σ2) / 2)
end


function Distributions.logpdf(d::ConditionalDists.BMN, x::Matrix{T}) where T<:Real
    n = size(d.μ,1)
    μ = mean(d)
    σ2 = var(d)
    _l(x, n, μ, σ2)
end

Zygote.@adjoint function _l(x, n, μ, σ2)
	_l(x, n, μ, σ2), Δ -> _∇l(Δ, x, n, μ, σ2)
end

μ = rand(2,10)
σ = rand(2,10)
d = ConditionalDists.BatchMvNormal(μ,σ)
x = rand(2,10)
f(x) = sum(logpdf(d,x))
df(x) = Zygote.gradient(f, x)[1]
ddf(x) = Zygote.gradient(x->sum(df(x)), x)[1]

df(x)
ddf(x)
