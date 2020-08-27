using ConditionalDists
using Flux

batch = 10000
xlength = 300
zlength = 150

f = Dense(zlength,xlength*2)
p = CMeanVarGaussian{DiagVar}(f)

z = randn(Float32, zlength, batch)
x = randn(Float32, xlength, batch)

@time logpdf(p, x, z)
@time logpdf(p, x, z)
@time logpdf(p, x, z)
