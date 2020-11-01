using Random
using Zygote

struct Normal
    μ
    σ
end

Random.rand(d::Normal) = randn()*d.σ + d.μ
f(a,b) = rand(Normal(a,b))
gs = Zygote.gradient(f,2,2)
