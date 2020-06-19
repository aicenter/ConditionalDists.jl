@testset "BatchMvNormal" begin

    xlen = 10
    batch = 3
    rtol = atol = 1e-6
    μ = rand(Float32, xlen, batch)
    σ = rand(Float32, xlen, batch)
    x = rand(Float32, xlen, batch)

    d = BatchMvNormal(μ,σ)
    @test length(d) == size(μ,1)
    @test eltype(d) == eltype(μ)
    @test Distributions.params(d) == (μ,σ)
    @test mean(d) == μ
    @test var(d) == σ .^2

    llh = logpdf(d,x)
    testds  = [MvNormal(m,s) for (m,s) in zip(eachcol(μ), eachcol(σ))]
    testllh = Float32.([logpdf(p,c) for (p,c) in zip(testds,eachcol(x))])

    @test all(isapprox.(llh, testllh, rtol=rtol, atol=atol))

    f(x) = sum(logpdf(d, x))
    reverse_zygote = Zygote.gradient(f, x)[1]
    reverse_diff = ReverseDiff.gradient(f, x)
    forward = ForwardDiff.gradient(f, x)

    @test isapprox(reverse_zygote, forward, rtol=rtol, atol=atol)
    @test isapprox(reverse_diff, forward, rtol=rtol, atol=atol)



    μ = rand(Float32, xlen, batch)
    s = rand(Float32, xlen)
    x = rand(Float32, xlen, batch)
    d = BatchMvNormal(μ,s)

    llh = logpdf(d,x)
    testds  = [MvNormal(m,s) for m in eachcol(μ)]
    testllh = Float32.([logpdf(p,c) for (p,c) in zip(testds,eachcol(x))])

    @test all(isapprox.(llh, testllh, rtol=rtol, atol=atol))

    f(x) = sum(logpdf(d, x))
    reverse_zygote = Zygote.gradient(f, x)[1]
    reverse_diff = ReverseDiff.gradient(f, x)
    forward = ForwardDiff.gradient(f, x)

    @test isapprox(reverse_zygote, forward, rtol=rtol, atol=atol)
    @test isapprox(reverse_diff, forward, rtol=rtol, atol=atol)

end
