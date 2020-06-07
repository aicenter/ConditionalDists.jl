@testset "TuringMvNormal" begin
    # Constructors

    μ = zeros(3)
    Σ = Array(Diagonal(ones(3)))
    σ = ones(3)
    @test TuringMvNormal(μ,Σ) isa ConditionalDists.TuringDenseMvNormal
    @test TuringMvNormal(μ,σ) isa ConditionalDists.TuringDiagMvNormal
    @test TuringMvNormal(μ,1.0) isa ConditionalDists.TuringScalMvNormal
    @test_throws MethodError TuringMvNormal(μ,1)

    @test TuringMvNormal(3,1.0) isa ConditionalDists.TuringScalMvNormal
    @test TuringMvNormal(σ) isa ConditionalDists.TuringDiagMvNormal
    @test TuringMvNormal(Σ) isa ConditionalDists.TuringDenseMvNormal
    @test TuringMvNormal(μ,2.0*I) isa ConditionalDists.TuringScalMvNormal

    d = TuringMvNormal(μ,σ)
    @test length(d) == length(μ)
    @test eltype(d) == eltype(μ)
    @test params(d) == (μ,σ)
    @test mean(d) == μ
    @test var(d) == σ
end
