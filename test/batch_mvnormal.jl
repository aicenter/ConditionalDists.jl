@testset "BatchMvNormal" begin

    xlen = 10
    batch = 3
    μ = rand(Float32, xlen, batch)
    σ = rand(Float32, batch)
    x = rand(Float32, xlen, batch)

    d = BatchMvNormal(μ,σ)

    llh = logpdf(d,x)
    
    testds  = [MvNormal(m,s) for (m,s) in zip(eachcol(μ),σ)]
    testllh = [logpdf(p,c) for (p,c) in zip(testds,eachcol(x))]

    @test eltype(llh) == Float32
    @test all(llh .≈ testllh)
end
