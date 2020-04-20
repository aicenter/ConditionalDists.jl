struct BatchMvNormal{Var<:AbstractVecOrMat,Mean<:AbstractMatrix}
    μ::Mean
    σ::Var
end

const BatchScalNormal = BatchMvNormal{<:AbstractVector}
const BatchDiagNormal = BatchMvNormal{<:AbstractMatrix}

length(d::BatchMvNormal) = size(d.μ,1)
eltype(d::BatchMvNormal) = eltype(d.μ)
mean(d::BatchMvNormal) = d.μ
var(d::BatchDiagNormal) = abs2.(d.σ)
var(d::BatchScalNormal) = fill!(similar(d.σ,size(d.μ,1)),1) .* reshape(abs2.(d.σ),1,:)

function rand(d::BatchScalNormal)
    μ = d.μ
    σ = d.σ
    r = randn!(similar(μ))
    μ .+ σ .* r
end

function logpdf(d::BatchMvNormal, x::AbstractMatrix)
    T = eltype(d)
    n = length(d)
    μ = mean(d)
    σ2 = var(d)
    -(vec(sum(((x - μ).^2) ./ σ2 .+ log.(σ2), dims=1)) .+ n*log(T(2π))) / 2
end
