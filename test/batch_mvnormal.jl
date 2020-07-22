@testset "BatchMvNormal" begin

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
    testds  = [MvNormal(m,s) for (m,s) in zip(eachcol(μs), eachcol(σs))]
    testllh = Float32.([logpdf(p,x) for (p,x) in zip(testds,eachcol(xs))])
    @test all(isapprox.(llh, testllh, rtol=rtol, atol=atol))

    f(x, μ, σ) = sum(logpdf(BatchMvNormal(μ, σ) |> gpu, x |> gpu))
    @test_nowarn Flux.gradient(f, xs, μs, σs)

    f(μ, σ) = sum(rand(BatchMvNormal(μ,σ) |> gpu))
    @test_nowarn Flux.gradient(f, μs, σs)

    # BatchScalMvNormal
    μs = rand(Float32, xlen, batch)
    ss = rand(Float32, batch)
    xs = rand(Float32, xlen, batch)
    d = BatchMvNormal(μs,ss)

    llh = logpdf(d|>gpu,xs|>gpu) |> cpu
    testds  = [MvNormal(m,s) for (m,s) in zip(eachcol(μs),ss)]
    testllh = Float32.([logpdf(p,x) for (p,x) in zip(testds,eachcol(xs))])
    @test all(isapprox.(llh, testllh, rtol=rtol, atol=atol))

    f(x, μ, σ) = sum(logpdf(BatchMvNormal(μ|>gpu, σ|>gpu), x|>gpu))
    @test_nowarn Flux.gradient(f, xs, μs, ss)

    f(μ, σ) = sum(rand(BatchMvNormal(μ,σ)|>gpu))
    @test_nowarn Flux.gradient(f, μs, ss)

end
