@testset "TuringMvNormal" begin
    # Constructors

    μ = zeros(3)
    Σ = Array(Diagonal(ones(3)))
    σ = ones(3)*2
    s = 2.0
    @test TuringMvNormal(μ,Σ) isa ConditionalDists.TuringDenseMvNormal
    @test TuringMvNormal(μ,σ) isa ConditionalDists.TuringDiagMvNormal
    @test TuringMvNormal(μ,s) isa ConditionalDists.TuringScalMvNormal
    @test_throws MethodError TuringMvNormal(μ,1)

    @test TuringMvNormal(3,s) isa ConditionalDists.TuringScalMvNormal
    @test TuringMvNormal(σ) isa ConditionalDists.TuringDiagMvNormal
    @test TuringMvNormal(Σ) isa ConditionalDists.TuringDenseMvNormal
    @test TuringMvNormal(μ,2.0*I) isa ConditionalDists.TuringScalMvNormal

    d = TuringMvNormal(μ,σ)
    @test length(d) == length(μ)
    @test eltype(d) == eltype(μ)
    @test params(d) == (μ,σ)
    @test mean(d) == μ
    @test var(d) == σ .^2

    x = rand(3)
    @test_broken logpdf(MvNormal(μ,Σ), x) ≈ logpdf(TuringMvNormal(μ,Σ), x)
    X = rand(3,10)
    @test_broken logpdf(MvNormal(μ,Σ), X) ≈ logpdf(TuringMvNormal(μ,Σ), X)

    x = rand(3)
    @test logpdf(MvNormal(μ,σ), x) ≈ logpdf(TuringMvNormal(μ,σ), x)
    X = rand(3,10)
    @test logpdf(MvNormal(μ,σ), X) ≈ logpdf(TuringMvNormal(μ,σ), X)

    x = rand(3)
    @test logpdf(MvNormal(μ,s), x) ≈ logpdf(TuringMvNormal(μ,s), x)
    X = rand(3,10)
    @test logpdf(MvNormal(μ,s), X) ≈ logpdf(TuringMvNormal(μ,s), X)
end
