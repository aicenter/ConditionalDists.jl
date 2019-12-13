@testset "CMeanVarGaussian" begin

    xlen = 3
    zlen = 2
    batch = 10
    T     = Float32

    # Test ScalarVar
    @testset "ScalarVar" begin
        p = CMeanVarGaussian{T,ScalarVar}(f32(Dense(zlen, xlen+1))) |> gpu

        z  = randn(T, zlen, batch) |> gpu
        μx = mean(p, z)
        σ2 = variance(p, z)
        @test mean_var(p, z) == (μx, σ2)
        @test size(μx) == (xlen, batch)
        @test size(σ2) == (1, batch)
        @test size(rand(p, z)) == (xlen, batch)

        x = randn(xlen, batch) |> gpu
        @test size(loglikelihood(p, x, z)) == (1, batch)

        x = randn(xlen) |> gpu
        z = randn(zlen) |> gpu
        @test size(loglikelihood(p, x, z)) == (1, )

        (h, w, c, n) = (4, 4, 2, batch)
        oc = 4
        p = CMeanVarGaussian{T,ScalarVar}(f32(Conv((3,3), c=>oc+1, pad = (1,1)))) |> gpu

        z = randn(T, h, w, c, n) |> gpu;
        μx = mean(p, z);
        σ2 = variance(p, z);
        @test mean_var(p, z) == (μx, σ2)
        @test size(μx) == (h, w, oc, batch)
        @test size(σ2) == (h, w, 1, batch)
        @test size(rand(p, z)) == (h, w, oc, batch)

        x = randn(h, w, oc, batch) |> gpu;
        @test size(loglikelihood(p, x, z)) == (1, 1, 1, batch) # i would like it more to be shape (1, batch)
    end

    @testset "DiagVar" begin
        p = CMeanVarGaussian{T,DiagVar}(f32(Dense(zlen, xlen*2))) |> gpu

        z  = randn(T, zlen, batch) |> gpu
        x  = randn(T, xlen, batch) |> gpu
        μx = mean(p, z)
        σ2 = variance(p, z)
        @test size(μx) == (xlen, batch)
        @test size(σ2) == (xlen, batch)
        @test size(rand(p, z)) == (xlen, batch)
        @test size(loglikelihood(p, x, z)) == (1, batch)

        (h, w, c, n) = (4, 4, 2, batch)
        oc = 4
        p = CMeanVarGaussian{T,DiagVar}(f32(Conv((3,3), c=>oc*2, pad = (1,1)))) |> gpu

        z = randn(T, h, w, c, n) |> gpu;
        μx = mean(p, z);
        σ2 = variance(p, z);
        @test mean_var(p, z) == (μx, σ2)
        @test size(μx) == (h, w, oc, batch)
        @test size(σ2) == (h, w, oc, batch)
        @test size(rand(p, z)) == (h, w, oc, batch)

        x = randn(h, w, oc, batch) |> gpu;
        @test size(loglikelihood(p, x, z)) == (1, 1, 1, batch)

        # Test show function
        msg = @capture_out show(p)
        @test occursin("CMeanVarGaussian", msg)
    end

end
