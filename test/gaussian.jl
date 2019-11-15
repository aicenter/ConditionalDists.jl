@testset "Gaussian" begin

    p  = Gaussian(zeros(2), ones(2)) |> gpu
    μ  = mean(p)
    σ2 = variance(p)
    @test mean_var(p) == (μ, σ2)
    @test size(rand(p, 10)) == (2, 10)
    @test size(loglikelihood(p, randn(2, 10)|>gpu)) == (1, 10)
    @test size(loglikelihood(p, randn(2)|>gpu)) == (1,)
    @test length(Flux.trainable(p)) == 2

    msg = @capture_out show(p)
    @test occursin("Gaussian", msg)

    μ = NoGradArray(zeros(2))
    p = Gaussian(μ, ones(2))
    @test length(Flux.trainable(p)) == 1
end
