using Test, Suppressor, Random
using Flux, Zygote

using Revise
using ConditionalDists

if Flux.use_cuda[] using CuArrays end

include("abstract_pdf.jl")
include("gaussian.jl")
include("cmean_gaussian.jl")
include("cmeanvar_gaussian.jl")
include("nogradarray.jl")
