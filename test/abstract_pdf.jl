@testset "Abstract PDFs" begin
    struct PDF <: ConditionalDists.AbstractPDF end
    struct CPDF <: ConditionalDists.AbstractCPDF end

    pdf = PDF()
    cpdf = CPDF()
    x = ones(1)

    @test_throws ErrorException mean_var(pdf)
    @test_throws ErrorException mean(pdf)
    @test_throws ErrorException variance(pdf)
    @test_throws ErrorException rand(pdf)
    @test_throws ErrorException loglikelihood(pdf, x)

    @test_throws ErrorException mean_var(cpdf, x)
    @test_throws ErrorException mean(cpdf, x)
    @test_throws ErrorException variance(cpdf, x)
    @test_throws ErrorException rand(cpdf, x)
    @test_throws ErrorException loglikelihood(cpdf, x, x)

end
