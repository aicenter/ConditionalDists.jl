@testset "Abstract PDFs" begin
    struct PDF{T<:Real} <: ConditionalDists.AbstractPDF{T} end
    struct CPDF{T<:Real} <: ConditionalDists.AbstractCPDF{T} end

    pdf = PDF{Float32}()
    cpdf = CPDF{Float32}()
    x = ones(1)

    @test_throws ErrorException mean_var(pdf)
    @test_throws ErrorException rand(pdf)
    @test_throws ErrorException loglikelihood(pdf, x)

    @test_throws ErrorException mean_var(cpdf, x)
    @test_throws ErrorException rand(cpdf, x)
    @test_throws ErrorException loglikelihood(cpdf, x, x)

end
