@testset "ConditionalMvNormal" begin

    xlength = 3
    zlength = 2
    batchsize = 10

    σvector(x::AbstractVector) = ones(Float32,xlength) .* 3
    σvector(x::AbstractMatrix) = ones(Float32,xlength,size(x,2)) .* 3
    σscalar(x::AbstractVector) = 2
    σscalar(x::AbstractMatrix) = ones(Float32,size(x,2)) .* 2

    x = rand(Float32, xlength) |> gpu
    z = rand(Float32, zlength) |> gpu
    X = rand(Float32, xlength, batchsize) |> gpu
    Z = rand(Float32, zlength, batchsize) |> gpu
    
    cases = [
        ("vector μ / vector σ",
         SplitLayer(zlength, [xlength,xlength], [identity,abs]), Vector, 4),
        ("vector μ / scalar σ",
         SplitLayer(zlength, [xlength,1], [identity,abs]), Real, 4),
        ("vector μ / fixed vector σ",
         SplitLayer(Dense(zlength,xlength), σvector), Vector, 2),
        ("vector μ / fixed scalar σ",
         SplitLayer(Dense(zlength,xlength), σscalar), Real, 2),
        ("vector μ / unit σ",
         Dense(zlength,xlength), Real, 2)
    ]

    disttypes(::Type{<:Vector}) = (TuringDiagMvNormal,ConditionalDists.BatchDiagMvNormal)
    disttypes(::Type{<:Real}) = (TuringScalMvNormal,ConditionalDists.BatchScalMvNormal)
    σsize(::Type{<:Vector}) = (xlength,)
    σsize(::Type{<:Real}) = ()


    for (name,mapping,T,nrps) in cases
        @testset "$name" begin
            p = ConditionalMvNormal(mapping) |> gpu
            (Texample,Tbatch) = disttypes(T)

            res = condition(p,z)
            μ = mean(res)
            σ2 = var(res)
            @test res isa Texample
            @test size(μ) == (xlength,)
            @test size(σ2) == σsize(T)

            loss() = logpdf(p,x,z)
            ps = Flux.params(p)
            @test length(ps) == nrps
            @test loss() isa Float32
            @test_nowarn Flux.gradient(loss, ps)

            f() = sum(rand(p,z))
            gs = Flux.gradient(f,ps)
            for p in ps
                g = gs[p]
                @test all(isfinite.(g)) && all(g .!= 0)
            end


            # batch tests
            res = condition(p,Z)
            μ = mean(res)
            σ2 = var(res)
            @test res isa Tbatch
            @test size(μ) == (xlength,batchsize)
            @test size(σ2) == (xlength,batchsize)

            loss() = sum(logpdf(p,X,Z))
            @test loss() isa Float32
            @test_nowarn Flux.gradient(loss, ps)

            f() = sum(rand(p,Z))
            gs = Flux.gradient(f,ps)
            for p in ps
                g = gs[p]
                @test all(isfinite.(g)) && all(g .!= 0)
            end
        end
    end
end
