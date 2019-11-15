module ConditionalDists

using Random
using Flux
using Zygote: @nograd

# functions that are overloaded by this module
import Base.length
import Random.rand
import Statistics.mean

# needed to make e.g. sampling work
@nograd similar, randn!, fill!

include("nogradarray.jl")

include("abstract_pdfs.jl")
include("gaussian.jl")
include("abstract_cgaussian.jl")
include("cmean_gaussian.jl")
include("cmeanvar_gaussian.jl")

end # module
