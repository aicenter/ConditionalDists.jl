@testset "CMeanGaussian" begin

    xlen  = 3
    zlen  = 2
    batch = 10
    T     = Float32

    @testset "DiagVar" begin
        mapping = Dense(zlen, xlen)
        v  = NoGradArray(ones(T, xlen))
        p  = CMeanGaussian{DiagVar}(mapping, v ) |> gpu
        z  = randn(T, zlen, batch) |> gpu
        μx = mean(p, z)
        σ2 = var(p, z)
        x  = rand(p, z)

        @test size(μx) == (xlen, batch)
        @test size(σ2) == (xlen, batch)
        @test size(x) == (xlen, batch)
        @test length(params(p)) == 2
        @test size(logpdf(p, x, z)) == (1, batch)

        # Test show function
        msg = @capture_out show(p)
        @test occursin("CMeanGaussian", msg)

        # test gradient
        loss() = sum(mean(p,z) .+ var(p,z))
        ps = params(p)
        gs = Flux.gradient(loss, ps)
        for _p in ps @test all(abs.(gs[_p]) .> 0) end
    end

    @testset "ScalarVar" begin
        mapping = Dense(zlen, xlen)
        v = ones(T, 1)

        p  = CMeanGaussian{ScalarVar}(mapping, v, xlen) |> gpu
        z  = randn(T, zlen, batch) |> gpu
        μx = mean(p, z)
        σ2 = var(p, z)
        x  = rand(p, z)

        @test size(μx) == (xlen, batch)
        @test size(σ2) == (xlen, batch)
        @test size(x) == (xlen, batch)
        @test length(params(p)) == 3
        @test size(logpdf(p, x, z)) == (1, batch)

        # test gradient
        loss() = sum(mean(p,z) .+ var(p,z))
        ps = params(p)
        gs = Flux.gradient(loss, ps)
        for _p in ps @test all(abs.(gs[_p]) .> 0) end
    end

end
