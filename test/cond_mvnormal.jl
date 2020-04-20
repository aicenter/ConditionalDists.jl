@testset "ConditionalMeanVarMvNormal" begin

    Flux.@functor ConditionalMeanVarMvNormal

    xlength = 3
    zlength = 2
    batchsize = 10
    m = Dense(zlength, 2*xlength)
    d = TuringDiagMvNormal(zeros(Float32,xlength), ones(Float32,xlength))
    p = ConditionalMeanVarMvNormal(d,m)

    res = condition(p, rand(zlength))
    μ = res.m
    σ2 = res.σ
    @test res isa TuringDiagMvNormal
    @test size(μ) == (xlength,)
    @test size(σ2) == (xlength,)

    res = condition(p, rand(zlength,batchsize))
    μ = mean(res)
    σ2 = var(res)
    @test res isa BatchMvNormal
    @test size(μ) == (xlength,batchsize)
    @test size(σ2) == (xlength,batchsize)

    x = rand(Float32, xlength)
    z = rand(Float32, zlength)
    loss() = logpdf(p,x,z)
    ps = Flux.params(p)
    @test length(ps) == 2
    @test_nowarn gs = Flux.gradient(loss, ps)

    x = rand(Float32, xlength, batchsize)
    z = rand(Float32, zlength, batchsize)
    loss() = sum(logpdf(p,x,z))
    ps = Flux.params(p)
    @test length(ps) == 2
    @test_nowarn gs = Flux.gradient(loss, ps)
end
