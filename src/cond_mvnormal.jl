struct ConditionalMvNormal{Tm} <: AbstractConditionalDistribution
    xlength::Int
    mapping::Tm
end

function condition(p::ConditionalMvNormal, z::AbstractVector)
    n = length(p)
    x = p.mapping(z)
    μ = x[1:n]
    σ = abs.(x[n+1:end])
    if length(σ) == 1
        σ = σ[1]
    end
    DistributionsAD.TuringMvNormal(μ,σ)  # for CuArrays/gradients
end

function condition(p::ConditionalMvNormal, z::AbstractMatrix)
    n = length(p)
    x = p.mapping(z)
    μ = x[1:n,:]
    σ = abs.(x[n+1:end,:])
    if size(σ,1) == 1
        σ = dropdims(σ, dims=1)
    end
    BatchMvNormal(μ,σ)
end

Base.length(p::ConditionalMvNormal) = p.xlength

# TODO: this should be moved to DistributionsAD
Distributions.mean(p::TuringDiagMvNormal) = p.m
Distributions.mean(p::TuringScalMvNormal) = p.m
Distributions.var(p::TuringDiagMvNormal) = p.σ .^2
Distributions.var(p::TuringScalMvNormal) = p.σ^2
