module ConditionalDists

using Random
using LinearAlgebra
using Distributions
using SpecialFunctions
using Flux
using Flux: @nograd

# functions that are overloaded by this module
import Base.length
import Base.eltype
import Random.rand
import Statistics.mean
import Distributions.cov
import Distributions.var
import Distributions.logpdf

abstract type AbstractConditionalDistribution end
const CMD = ContinuousMultivariateDistribution
const ACD = AbstractConditionalDistribution

export Gaussian
export InverseGamma

export CMeanGaussian, CMGaussian
export CMeanVarGaussian, CMVGaussian
export AbstractVar, DiagVar, ScalarVar

export mean
export cov
export var
export mean_var
export rand
export logpdf

# needed to make e.g. sampling work
@nograd similar, randn!, fill!

include("nogradarray.jl")

include("gaussian.jl")
include("abstract_cgaussian.jl")
include("cmean_gaussian.jl")
include("cmeanvar_gaussian.jl")
include("constspec_gaussian.jl")

include("inverse_gamma.jl")
# include("inverse_wishart.jl")

end # module
