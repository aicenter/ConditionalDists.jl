@testset "CMeanGaussian" begin

    xlen  = 3
    zlen  = 2
    batch = 10
    T     = Float32

    for (Var, vlen) in [(ScalarVar, 1), (DiagVar, xlen)]
        @testset "$Var" begin
            mapping = Dense(zlen, xlen)
            v  = NoGradArray(ones(T, vlen))
            p  = CMeanGaussian{Var}(mapping, v, xlen) |> gpu
            z  = randn(T, zlen, batch) |> gpu
            μx = mean(p, z)
            σ2 = var(p, z)
            Σ  = cov(p, z)
            x  = rand(p, z)

            @test eltype(p) == T
            @test length(p) == xlen
            @test size(μx) == (xlen, batch)
            @test size(σ2) == (xlen, batch)
            @test size(x) == (xlen, batch)
            @test length(params(p)) == 2
            @test size(logpdf(p, x, z)) == (1, batch)

            # Covariance test
            # _σ2 = Σ .* collect(eachcol(ones(T,xlen,batch)|>gpu))
            # _σ2 = hcat(_σ2...)
            #@test all(_σ2 .== σ2)

            # Test show function
            msg = sprint(show, p)
            @test occursin("CMeanGaussian", msg)

            # test gradient
            loss() = sum(logpdf(p,x,z))
            ps = params(p)
            gs = Flux.gradient(loss, ps)
            for _p in ps @test all(abs.(gs[_p]) .> 0) end
        end
    end
end
