using Test
using Distributions: logpdf
using DistributionsAD
using ConditionalDists
using Flux


@testset "Conditional MvNormals with mean and variance from mapping" begin

    Flux.@functor ConditionalMeanVarMvNormal

    m = f64(Dense(2,6))
    d = ConditionalMeanVarMvNormal(m, 3)
    x = rand(3)
    z = rand(2)

    loss() = logpdf(d,x,z)
    ps = Flux.params(d)
    @test length(ps) == 2
    @test_nowarn gs = Flux.gradient(loss, ps)
end
      
@testset "Conditional MvNormal with mean from mapping and shared variance" begin

    Flux.@functor ConditionalMeanMvNormal
    # need to call functor on the Turing Normal to learn the variance
    Flux.@functor TuringDiagMvNormal

    m = f64(Dense(2,3))
    d = ConditionalMeanMvNormal(m, ones(3))
    x = rand(3)
    z = rand(2)
    
    loss() = logpdf(d,x,z)
    ps = Flux.params(d)
    gs = Flux.gradient(loss, ps)

    @test length(ps) == 4
    @test gs[d.d.m] == nothing

    # calling functor on the Normal results in one parameter too much (the
    # mean, which comes from the mapping).
    #
    # Also I would like a mechanism that lets me specify that certain
    # parameters should be constant, e.g. to have a fixed MvNormal prior in a
    # model that I can call `Flux.params` on.
    #
    # Currently looking into this via Chaincutters.jl (we have been doing it so
    # far with a custom `NoGradArray` type which I don't think is the nicest
    # solution...)
    # The use case would be e.g. something like this:
    #
    # struct VAE
    #     prior::DiagMvNormal
    #     encoder::ConditionalMeanVarMvNormal
    #     decoder::ConditionalMeanVarMvNormal
    # end
    #
    # function elbo(m::VAE, x; β=1)
    #     de = condition(m.encoder, x)
    #     z  = rand(de)
    #     llh = logpdf(m.decoder, x, z)
    #     kld = kl_divergence(de, m.prior)
    #     llh - β*kld
    # end
    #
    # for this it would of course be great to have an efficient way of doing
    # all of the above with batches. we are currently doing this (without using
    # DAD.jl). I am not sure what the smartest approach is. Just working with
    # vectors of distributions would sacrifice some speed because that would not
    # use fast matmuls I guess?
 
end
