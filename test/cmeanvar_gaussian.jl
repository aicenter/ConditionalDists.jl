@testset "CMeanVarGaussian" begin

    xlen = 3
    zlen = 2
    batch = 10
    T     = Float32

    for (Var, outlen) in [(ScalarVar, xlen+1), (DiagVar, xlen*2)]
        @testset "ScalarVar" begin
            p = CMeanVarGaussian{Var}(f32(Dense(zlen, outlen))) |> gpu

            z  = randn(T, zlen, batch) |> gpu
            μx = mean(p, z)
            σ2 = var(p, z)
            Σ  = cov(p,z)

            @test eltype(p) == T
            @test length(Σ) == batch
            @test size(Σ[1]) == (xlen, xlen)
            @test length(p,z[:,1]) == xlen
            @test length(p,z) == xlen
            @test mean_var(p, z) == (μx, σ2)
            @test size(μx) == (xlen, batch)
            @test size(σ2) == (xlen, batch)
            @test size(rand(p, z)) == (xlen, batch)
            @test length(params(p)) == 2

            x = randn(T, xlen, batch) |> gpu
            @test size(logpdf(p, x, z)) == (1, batch)

            x = randn(xlen) |> gpu
            z = randn(zlen) |> gpu
            @test size(logpdf(p, x, z)) == (1, 1)

            # Test show function
            msg = @capture_out show(p)
            @test occursin("CMeanVarGaussian", msg)
        end
    end
end
