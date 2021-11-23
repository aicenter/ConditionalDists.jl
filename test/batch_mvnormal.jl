@testset "BatchMvNormal" begin

    # for the BatchMvNormal tests to work BatchMvNormals have to be functors!
    Flux.@functor ConditionalDists.BatchDiagMvNormal
    Flux.@functor ConditionalDists.BatchScalMvNormal

    # BatchDiagMvNormal
    xlen = 10
    batch = 3
    rtol = atol = 1e-6
    μs = rand(Float32, xlen, batch)
    σs = rand(Float32, xlen, batch)
    xs = rand(Float32, xlen, batch)

    d = BatchMvNormal(μs,σs)
    @test eltype(d) == eltype(μs)
    @test Distributions.params(d) == (μs,σs)
    @test all(mean(d) .== μs)
    @test all(var(d) .== σs .^2)

    llh = logpdf(d|>gpu, xs|>gpu) |> cpu
    testds  = [MvNormal(m,Diagonal(s)) for (m,s) in zip(eachcol(μs), eachcol(σs))]
    testllh = Float32.([logpdf(p,x) for (p,x) in zip(testds,eachcol(xs))])
    @test all(isapprox.(llh, testllh, rtol=rtol, atol=atol))

    f1(x, μ, σ) = sum(logpdf(BatchMvNormal(μ, σ) |> gpu, x |> gpu))
    @test_nowarn Flux.gradient(f1, xs, μs, σs)

    f1(μ, σ) = sum(rand(BatchMvNormal(μ,σ) |> gpu))
    @test_nowarn Flux.gradient(f1, μs, σs)

    # BatchScalMvNormal
    μs = rand(Float32, xlen, batch)
    ss = rand(Float32, batch)
    xs = rand(Float32, xlen, batch)
    d = BatchMvNormal(μs,ss)

    llh = logpdf(d|>gpu,xs|>gpu) |> cpu
    testds  = [MvNormal(m,s) for (m,s) in zip(eachcol(μs),ss)]
    testllh = Float32.([logpdf(p,x) for (p,x) in zip(testds,eachcol(xs))])
    @test all(isapprox.(llh, testllh, rtol=rtol, atol=atol))

    f3(x, μ, σ) = sum(logpdf(BatchMvNormal(μ|>gpu, σ|>gpu), x|>gpu))
    @test_nowarn Flux.gradient(f3, xs, μs, ss)

    f4(μ, σ) = sum(rand(BatchMvNormal(μ,σ)|>gpu))
    @test_nowarn Flux.gradient(f4, μs, ss)

end
