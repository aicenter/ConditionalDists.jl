@testset "ConditionalMeanVarMvNormal" begin

    Flux.@functor ConditionalMeanVarMvNormal

    xlength = 3
    zlength = 2
    batchsize = 10
    m = Dense(zlength, 2*xlength)
    d = TuringMvNormal(zeros(Float32,xlength), ones(Float32,xlength))
    p = ConditionalMeanVarMvNormal(d,m)

    res = condition(p, rand(zlength))
    μ = mean(res)
    σ2 = var(res)
    @test res isa ConditionalDists.TuringDiagMvNormal
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
    loss(x) = sum(logpdf(p,x,z))
    ps = Flux.params(p)

    reverse_zygote = Zygote.gradient(loss, x)[1]
    reverse_diff = ReverseDiff.gradient(loss, x)
    forward = ForwardDiff.gradient(loss, x)

    rtol = atol = 1e-7
    #@test isapprox(reverse_tracker, forward, rtol=rtol, atol=atol)
    @test isapprox(reverse_zygote, forward, rtol=rtol, atol=atol)
    @test isapprox(reverse_diff, forward, rtol=rtol, atol=atol)

end
