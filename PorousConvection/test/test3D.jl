include("../scripts/PorousConvection_3D_xpu.jl")

testing  = true
do_check = true
T_final_3D = porous_convection_3D(;do_check, testing)

# Unit Test
@testset "Average Function Tests - 3D" begin
    @test av1([[10, 20, 30] [40, 50, 60] [70, 80, 90]]) == [[10, 10] [10, 10] [10, 10]]
    @test avx([[10, 20, 30] [40, 50, 60] [70, 80, 90]]) == [[10, 10] [40, 50, 60] [70, 80, 90]]
    @test avy([[10, 20, 30] [40, 50, 60] [70, 80, 90]]) == [[10, 20, 30] [10, 10] [70, 80, 90]]
    @test avz([[10, 20, 30] [40, 50, 60] [70, 80, 90]]) == [[10, 20, 30] [40, 50, 60] [10, 10]]
end

# Reference Test
inds        = sort(rand(1:length(T_final_3D), 50))
inds_linear = LinearIndices(inds)
T_vals      = T_final_3D[inds_linear]

@testset "PorousConvection3D" begin
    # Write your tests here.
    @test all(T_vals .≈ T_final_3D[inds_linear])
end