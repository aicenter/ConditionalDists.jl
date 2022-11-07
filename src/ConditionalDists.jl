module ConditionalDists

using LinearAlgebra
using Random
using Distributions
using DistributionsAD
using ChainRulesCore
using Functors
#using Requires
using Flux

export condition

export ConditionalDistribution
export ConditionalMvNormal
export SplitLayer

include("cond_dist.jl")

include("batch_mvnormal.jl")
include("cond_mvnormal.jl")
include("utils.jl")


end # module
