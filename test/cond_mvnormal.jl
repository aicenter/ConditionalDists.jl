@testset "ConditionalMeanVarMvNormal" begin

    Flux.@functor ConditionalMeanVarMvNormal

    xlength = 3
    zlength = 2
    batchsize = 10
    m = Dense(zlength, 2*xlength)
    d = MvNormal(zeros(Float32,xlength), ones(Float32,xlength))
    p = ConditionalMeanVarMvNormal(d,m)

    res = condition(p, rand(zlength))
    μ = mean(res)
    Σ = cov(res)
    σ2 = var(res)
    display(res)
    @test_broken Σ isa PDMats.PDiagMat
    @test size(μ) == (xlength,)
    @test size(σ2) == (xlength,)

    x = rand(Float32, xlength)
    z = rand(Float32, zlength)
    loss(x) = logpdf(p,x,z)
    @test_broken loss(x) isa Float32

    zygote = Zygote.gradient(loss, x)[1]
    reverse = ReverseDiff.gradient(loss, x)
    forward = ForwardDiff.gradient(loss, x)
    tracker = Tracker.data(Tracker.gradient(loss, x)[1])
    finitediff = FiniteDifferences.grad(central_fdm(5, 1), loss, x)[1]

    rtol = atol = 1e-6
    @test_broken all(isapprox.(zygote,  finitediff, rtol=rtol, atol=atol))
    @test_broken all(isapprox.(forward, finitediff, rtol=rtol, atol=atol))
    @test_broken all(isapprox.(reverse, finitediff, rtol=rtol, atol=atol))
    @test_broken all(isapprox.(tracker, finitediff, rtol=rtol, atol=atol))

    #=
    res = condition(p, rand(zlength,batchsize))
    μ = mean(res)
    σ2 = var(res)
    @test res isa BatchMvNormal
    @test size(μ) == (xlength,batchsize)
    @test size(σ2) == (xlength,batchsize)

    ps = Flux.params(p)
    @test length(ps) == 2
    @test_nowarn gs = Flux.gradient(loss, ps)

    x = rand(Float32, xlength, batchsize)
    z = rand(Float32, zlength, batchsize)
    loss(x) = sum(logpdf(p,x,z))
    ps = Flux.params(p)

    rtol = atol = 1e-7
    #@test isapprox(reverse_tracker, forward, rtol=rtol, atol=atol)
    @test isapprox(reverse_zygote, forward, rtol=rtol, atol=atol)
    @test isapprox(reverse_diff, forward, rtol=rtol, atol=atol)
    =#

end
