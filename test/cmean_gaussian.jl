@testset "CMeanGaussian" begin

    xlen  = 3
    zlen  = 2
    batch = 10
    T     = Float32

    for Var in [ScalarVar, DiagVar]
        @testset "$Var" begin
            mapping = Dense(zlen, xlen)
            v  = NoGradArray(ones(T, xlen))
            p  = CMeanGaussian{Var}(mapping, v, xlen) #|> gpu
            z  = randn(T, zlen, batch) #|> gpu
            μx = mean(p, z)
            σ2 = var(p, z)
            Σ  = cov(p, z)
            x  = rand(p, z)

            @test eltype(p) == T
            @test all(first(cov(p,z)) * ones(T,3) .== σ2[:,1])
            @test length(p) == xlen
            @test size(μx) == (xlen, batch)
            @test size(σ2) == (xlen, batch)
            @test size(x) == (xlen, batch)
            @test length(params(p)) == 2
            @show logpdf(p,x,z)
            @test size(logpdf(p, x, z)) == (batch,)

            # Test show function
            msg = sprint(show, p)
            @test occursin("CMeanGaussian", msg)

            # test gradient
            loss() = sum(mean(p,z) .+ var(p,z))
            ps = params(p)
            gs = Flux.gradient(loss, ps)
            for _p in ps @test all(abs.(gs[_p]) .> 0) end
        end
    end
end
