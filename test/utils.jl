@testset "SplitLayer" begin
    l = SplitLayer(x->x .+ 1, _->1)
    x = rand(3)
    (a,b) = l(x)
    @test all(a .≈ x .+ 1)
    @test b == 1

    l = SplitLayer(3,[2,4])
    (a,b) = l(x)
    @test size(a) == (2,)
    @test size(b) == (4,)
end
