@testset "ConditionalMvNormal" begin

    Flux.@functor ConditionalMvNormal

    xlength = 3
    zlength = 2
    batchsize = 10
    m = Dense(zlength, 2*xlength)
    d = MvNormal(zeros(Float32,xlength), ones(Float32,xlength))
    p = ConditionalMvNormal(d,m)

    # MvNormal
    res = condition(p, rand(zlength))
    μ = mean(res)
    σ2 = var(res)
    @test res.Σ isa PDMats.PDiagMat
    @test size(μ) == (xlength,)
    @test size(σ2) == (xlength,)

    x = rand(Float32, xlength)
    z = rand(Float32, zlength)
    loss() = logpdf(p,x,z)
    ps = Flux.params(p)
    @test_broken loss() isa Float32
    @test_nowarn Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_nowarn Flux.gradient(f, ps)

    # BatchDiagMvNormal
    res = condition(p, rand(zlength,batchsize))
    μ = mean(res)
    σ2 = var(res)
    @test res isa ConditionalDists.BatchDiagMvNormal
    @test size(μ) == (xlength,batchsize)
    @test size(σ2) == (xlength,batchsize)

    x = rand(Float32, xlength, batchsize)
    z = rand(Float32, zlength, batchsize)
    loss() = sum(logpdf(p,x,z))
    ps = Flux.params(p)
    @test length(ps) == 2
    @test loss() isa Float32
    @test_nowarn gs = Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_nowarn Flux.gradient(f, ps)


    # BatchScalMvNormal
    m = Dense(zlength, xlength+1)
    d = MvNormal(zeros(Float32,xlength), 1f0)
    p = ConditionalMvNormal(d,m)

    res = condition(p, rand(zlength,batchsize))
    μ = mean(res)
    σ2 = var(res)
    @test res isa ConditionalDists.BatchScalMvNormal
    @test size(μ) == (xlength,batchsize)
    @test size(σ2) == (batchsize,)

    x = rand(Float32, xlength, batchsize)
    z = rand(Float32, zlength, batchsize)
    loss() = sum(logpdf(p,x,z))
    ps = Flux.params(p)
    @test length(ps) == 2
    @test loss() isa Float32
    @test_nowarn gs = Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_nowarn Flux.gradient(f, ps)

end
