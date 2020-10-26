@testset "ConditionalMvNormal" begin

    xlength = 3
    zlength = 2
    batchsize = 10
    m = SplitLayer(zlength, [xlength,xlength], [identity,abs])
    p = ConditionalMvNormal(m) |> gpu
    x = rand(Float32, xlength) |> gpu
    z = rand(Float32, zlength) |> gpu
    X = rand(Float32, xlength, batchsize) |> gpu
    Z = rand(Float32, zlength, batchsize) |> gpu

    # MvNormal
    res = condition(p,z)
    μ = mean(res)
    σ2 = var(res)
    @test res isa TuringDiagMvNormal
    @test size(μ) == (xlength,)
    @test size(σ2) == (xlength,)

    loss() = logpdf(p,x,z)
    ps = Flux.params(p)
    @test_broken loss() isa Float32
    @test_nowarn Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_broken Flux.gradient(f, ps)

    # BatchDiagMvNormal
    res = condition(p,Z)
    μ = mean(res)
    σ2 = var(res)
    @test res isa ConditionalDists.BatchDiagMvNormal
    @test size(μ) == (xlength,batchsize)
    @test size(σ2) == (xlength,batchsize)

    loss() = sum(logpdf(p,X,Z))
    ps = Flux.params(p)
    @test length(ps) == 4
    @test loss() isa Float32
    @test_nowarn gs = Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_nowarn Flux.gradient(f, ps)


    # BatchScalMvNormal
    m = SplitLayer(zlength, [xlength,1])
    p = ConditionalMvNormal(m) |> gpu

    res = condition(p,Z)
    μ = mean(res)
    σ2 = var(res)
    @test res isa ConditionalDists.BatchScalMvNormal
    @test size(μ) == (xlength,batchsize)
    @test size(σ2) == (xlength,batchsize)

    loss() = sum(logpdf(p,X,Z))
    ps = Flux.params(p)
    @test length(ps) == 4
    @test loss() isa Float32
    @test_nowarn gs = Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_nowarn Flux.gradient(f, ps)


    # Unit variance
    m = Dense(zlength,xlength)
    p = ConditionalMvNormal(m) |> gpu

    res = condition(p,Z)
    μ = mean(res)
    σ2 = var(res)
    @test res isa ConditionalDists.BatchScalMvNormal
    @test size(μ) == (xlength,batchsize)
    @test size(σ2) == (xlength,batchsize)

    loss() = sum(logpdf(p,X,Z))
    ps = Flux.params(p)
    @test length(ps) == 2
    @test loss() isa Float32
    @test_nowarn gs = Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_nowarn Flux.gradient(f, ps)


    # Fixed scalar variance
    m = Dense(zlength,xlength)
    σ(x::AbstractVector) = 2
    σ(x::AbstractMatrix) = ones(Float32,size(x,2)) .* 2
    p = ConditionalMvNormal(SplitLayer(m,σ)) |> gpu

    res = condition(p,Z)
    μ = mean(res)
    σ2 = var(res)
    @test res isa ConditionalDists.BatchScalMvNormal
    @test size(μ) == (xlength,batchsize)
    @test size(σ2) == (xlength,batchsize)

    loss() = sum(logpdf(p,X,Z))
    ps = Flux.params(p)
    @test length(ps) == 2
    @test loss() isa Float32
    @test_nowarn gs = Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_nowarn Flux.gradient(f, ps)


    # Shared scalar variance TODO: add case for vector shared variance
    m = Dense(zlength,xlength)
    # σ = ones(Float32,xlength) .* 2
    σ = 2f0
    p = ConditionalMvNormal(SplitLayer(m,σ)) |> gpu

    res = condition(p,Z)
    μ = mean(res)
    σ2 = var(res)
    @test all(σ2 .== σ .^2)
    @test res isa ConditionalDists.BatchScalMvNormal
    @test size(μ) == (xlength,batchsize)
    @test size(σ2) == (xlength,batchsize)

    loss() = sum(logpdf(p,X,Z))
    ps = Flux.params(p)
    @test length(ps) == 3
    @test loss() isa Float32
    @test_nowarn gs = Flux.gradient(loss, ps)

    f() = sum(rand(p,z))
    @test_nowarn Flux.gradient(f, ps)

end
