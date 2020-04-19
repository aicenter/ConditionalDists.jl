using ConditionalDists
using DistributionsAD
using Distributions: logpdf
using Flux

batch = 10000
xlength = 300
zlength = 150

f = Dense(zlength,xlength*2)
d = TuringDiagMvNormal(zeros(Float32,xlength), ones(Float32,xlength))
p = ConditionalMeanVarMvNormal(f,d)

z = randn(Float32, zlength, batch)
x = randn(Float32, xlength, batch)

@time logpdf(p, x, z)
@time logpdf(p, x, z)
@time logpdf(p, x, z)

@info "done"
