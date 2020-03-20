@testset "ConstSpecGaussian" begin

    xlen  = 3
    zlen  = 2
    batch = 10
    T     = Float32

    pdf = Gaussian(zeros(T,xlen), ones(T,xlen)) |> gpu

    mapping = Dense(zlen, xlen)
    v = NoGradArray(ones(T, xlen))
    cpdf = CMeanGaussian{DiagVar}(mapping, v) |> gpu

    p = ConstSpecGaussian(pdf, cpdf)
    z  = randn(T, zlen, batch) |> gpu

    μc = const_mean(p)
    σc = const_var(p)
    @test size(μc) == (xlen,)
    @test size(μc) == (xlen,)

    μs = spec_mean(p, z)
    σs = spec_var(p, z)
    @test size(μs) == (xlen,batch)
    @test size(σs) == (xlen,batch)

    _μc, _μs = mean(p,z)
    @test all(μc .== _μc)
    @test all(μs .== _μs)
    _σc, _σs = var(p,z)
    @test all(σc .== _σc)
    @test all(σs .== _σs)

    (c,s) = rand(p, z)
    x = c .+ s
    @test length(params(p)) == 4
    @test size(logpdf(p,x,z)) == (1,batch)

    # Test show function
    msg = summary(p)
    @test occursin("ConstSpecGaussian", msg)
end
