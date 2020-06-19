@testset "TuringMvNormal" begin
    μ = rand(Float32,3)
    Σ = Array(Diagonal(ones(Float32,3)))
    σ = ones(Float32,3)*2
    s = 2.0f0
    @test TuringMvNormal(μ,Σ) isa ConditionalDists.TuringDenseMvNormal
    @test TuringMvNormal(μ,σ) isa ConditionalDists.TuringDiagMvNormal
    @test TuringMvNormal(μ,s) isa ConditionalDists.TuringScalMvNormal
    @test_throws MethodError TuringMvNormal(μ,1)

    @test TuringMvNormal(3,s) isa ConditionalDists.TuringScalMvNormal
    @test TuringMvNormal(σ) isa ConditionalDists.TuringDiagMvNormal
    @test TuringMvNormal(Σ) isa ConditionalDists.TuringDenseMvNormal
    @test TuringMvNormal(μ,2.0f0*I) isa ConditionalDists.TuringScalMvNormal

    d = TuringMvNormal(μ,σ)
    @test length(d) == length(μ)
    @test eltype(d) == eltype(μ)
    @test Distributions.params(d) == (μ,σ)
    @test mean(d) == μ
    @test var(d) == σ .^2

    rtol = atol = 1e-7

    for x in [rand(Float32,3), rand(Float32,3,10)]
        for s in [s,σ,Σ]
            standard_d = MvNormal(μ,s)
            turing_d = TuringMvNormal(μ,s)
            @test isapprox(Float32.(logpdf(standard_d, x)), logpdf(turing_d, x), rtol=rtol, atol=atol)

            f(x) = sum(logpdf(turing_d, x))
            @test f(x) isa Float32

            #reverse_tracker = Tracker.data(Tracker.gradient(f, x)[1])
            reverse_zygote = Zygote.gradient(f, x)[1]
            reverse_diff = ReverseDiff.gradient(f, x)
            forward = ForwardDiff.gradient(f, x)

            #@test isapprox(reverse_tracker, forward, rtol=rtol, atol=atol)
            @test isapprox(reverse_zygote, forward, rtol=rtol, atol=atol)
            @test isapprox(reverse_diff, forward, rtol=rtol, atol=atol)
        end
    end
end
