include("../scripts/PorousConvection_2D_xpu.jl")
using Test
using PorousConvection

testing  = true
do_check = true
T_final_2D = porous_convection_2D(;do_check, testing)

# Unit Test
@testset "Average Function Tests - 2D" begin
    @test av1([[10, 20, 30, 40] [50, 60, 70, 80]]) == [15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]
    @test avx([[10, 20, 30, 40] [50, 60, 70, 80]]) == [15.0 55.0; 25.0 65.0; 35.0 75.0]
    @test avy([[10, 20, 30, 40] [50, 60, 70, 80]]) == [30.0; 40.0; 50.0; 60.0;;]
end

# Reference Test
inds        = sort(rand(1:length(T_final_2D), 50))
inds_linear = LinearIndices(inds)
T_vals      = T_final_2D[inds_linear]

@testset "PorousConvection2D" begin
    # Write your tests here.
    @test all(T_vals .≈ T_final_2D[inds_linear])
end