using Test
using Suppressor
using Random
using Flux
using ConditionalDists

if Flux.use_cuda[] using CuArrays end

include("abstract_pdf.jl")
include("gaussian.jl")
include("cmean_gaussian.jl")
include("cmeanvar_gaussian.jl")
include("constspec_gaussian.jl")
include("nogradarray.jl")
