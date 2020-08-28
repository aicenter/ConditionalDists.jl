module ConditionalDists

using LinearAlgebra
using Random
using StatsBase
using Distributions
using DistributionsAD

export condition

export ConditionalDistribution
export ConditionalMvNormal

include("cond_dist.jl")

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")

using Requires
function __init__()
    @require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" include("flux.jl")
end

end # module
