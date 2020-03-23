@testset "CMeanVarGaussian" begin

    xlen = 3
    zlen = 2
    batch = 10
    T     = Float32

    for (Var, outlen) in [(ScalarVar, xlen+1), (DiagVar, xlen*2)]
        @testset "$Var" begin
            p = CMeanVarGaussian{Var}(f32(Dense(zlen, outlen))) |> gpu

            z  = randn(T, zlen, batch) |> gpu
            μx = mean(p, z)
            σ2 = var(p, z)
            Σ  = cov(p, z)

            @test eltype(p) == T
            @test length(Σ) == batch
            @test length(p,z[:,1]) == xlen
            @test length(p,z) == xlen
            @test mean_var(p, z) == (μx, σ2)
            @test size(μx) == (xlen, batch)
            @test size(σ2) == (xlen, batch)
            @test size(rand(p, z)) == (xlen, batch)
            @test length(params(p)) == 2

            # covariance test
            # _σ2 = Σ .* collect(eachcol(ones(T,xlen,batch)|>gpu))
            # _σ2 = hcat(_σ2...)
            #@test all(_σ2 .== σ2)

            x = randn(T, xlen, batch) |> gpu
            @test size(logpdf(p, x, z)) == (1, batch)

            x = randn(T, xlen) |> gpu
            z = randn(T, zlen) |> gpu
            @test size(logpdf(p, x, z)) == (1,1)

            # Test show function
            msg = sprint(show, p)
            @test occursin("CMeanVarGaussian", msg)

            if Var == ScalarVar
                z  = randn(T, zlen, batch) |> gpu
                σ2 = ConditionalDists.svar(p,z)
                @test size(σ2) == (1, batch)
                @test all(ConditionalDists.mean_svar(p,z)[2] .== σ2)
                if !(Flux.use_cuda[])
                    @test first(Σ) isa UniformScaling
                end
            else
                @test size(first(Σ)) == (xlen, xlen)
            end

            x = randn(T, xlen, batch) |> gpu
            z = randn(T, zlen, batch) |> gpu
            loss() = sum(logpdf(p,x,z))
            ps = params(p)
            gs = Flux.gradient(loss, ps)
            for _p in ps @test all(abs.(gs[_p]) .> 0) end
        end
    end
end
