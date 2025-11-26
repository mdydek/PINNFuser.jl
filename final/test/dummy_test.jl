using Test

@testset "MyPackage Tests" begin
    @test 1 + 1 == 2

    @test 0.1 + 0.2 ≈ 0.3
end
