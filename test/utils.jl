@testset "SplitLayer" begin
    # constant variance
    l = SplitLayer(x->x .+ 1, _->1)
    x = rand(3)
    (a,b) = l(x)
    @test all(a .≈ x .+ 1)
    @test b == 1

    l = SplitLayer(3,[2,4])
    (a,b) = l(x)
    @test size(a) == (2,)
    @test size(b) == (4,)

    # shared but learned variance (vector)
    l = SplitLayer(x->x, ones(4))
    (a,b) = l(x)
    @test size(a) == (3,)
    @test size(b) == (4,)

    (a,b) = l(rand(3,10))
    @test size(a) == (3,10)
    @test size(b) == (4,10)

    # shared but learned variance (scalar)
    l = SplitLayer(x->x,1)
    (a,b) = l(x)
    @test size(a) == (3,)
    @test size(b) == ()

    (a,b) = l(rand(3,10))
    @test size(a) == (3,10)
    @test size(b) == (10,)
end
