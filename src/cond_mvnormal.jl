struct ConditionalMvNormal{Tm} <: AbstractConditionalDistribution
    mapping::Tm
end

function condition(p::ConditionalMvNormal, z::AbstractVector)
    (μ,σ) = p.mapping(z)
    if length(σ) == 1
        σ = σ[1]
    end
    DistributionsAD.TuringMvNormal(μ,σ)  # for CuArrays/gradients
end

function condition(p::ConditionalMvNormal, z::AbstractMatrix)
    (μ,σ) = p.mapping(z)
    if size(σ,1) == 1
        σ = dropdims(σ, dims=1)
    end
    BatchMvNormal(μ,σ)
end

# TODO: this should be moved to DistributionsAD
Distributions.mean(p::TuringDiagMvNormal) = p.m
Distributions.mean(p::TuringScalMvNormal) = p.m
Distributions.var(p::TuringDiagMvNormal) = p.σ .^2
Distributions.var(p::TuringScalMvNormal) = p.σ^2
