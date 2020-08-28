@testset "ConditionalDistribution" begin

    xlength = 3
    zlength = 2
    batchsize = 10
    m = SplitLayer(zlength, [xlength,xlength], [identity,abs])
    d = TuringMvNormal
    p = ConditionalDistribution(d,m) |> gpu

    # MvNormal
    res = condition(p, rand(zlength) |> gpu)
    μ = mean(res)
    σ2 = var(res)
    @test res isa TuringDiagMvNormal
    @test size(μ) == (xlength,)
    @test size(σ2) == (xlength,)

    x = rand(Float32, xlength) |> gpu
    z = rand(Float32, zlength) |> gpu
    loss() = logpdf(p,x,z)
    ps = Flux.params(p)
    @test_broken loss() isa Float32
    @test_nowarn Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_nowarn Flux.gradient(f, ps)

    # Array of DiagMvNormal
    res = condition(p, rand(zlength,batchsize)|>gpu)
    @test res isa DistributionsAD.VectorOfMultivariate

    x = rand(Float32, xlength, batchsize) |> gpu
    z = rand(Float32, zlength, batchsize) |> gpu
    loss() = sum(logpdf(p,x,z))
    ps = Flux.params(p)
    @test length(ps) == 4
    @test_broken loss() isa Float32
    @test_nowarn gs = Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_nowarn Flux.gradient(f, ps)

end
