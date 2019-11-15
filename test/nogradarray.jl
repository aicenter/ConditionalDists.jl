@testset "NoGradArray" begin
    x = NoGradArray(ones(3))
    y = ones(3)

    p = Gaussian(x, y)
    @test length(params(p)) == 1

    @testset "Params on CPU/GPU" begin
        if Flux.use_cuda[]
            g = gpu(p)
            @test length(params(g)) == 1
            @test rand(g) isa CuArray

            c = cpu(g)
            @test length(params(c)) == 1
            @test rand(c) isa Array
        end
    end

    @testset "Gradients" begin
        loss() = sum(p.σ)
        ps = params(p)

        gs = Zygote.gradient(loss, ps)
        for p in ps
            @test haskey(gs, p)
        end
    end

end
