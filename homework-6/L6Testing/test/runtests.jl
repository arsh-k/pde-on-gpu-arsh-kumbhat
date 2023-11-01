include("../scripts/diffusion_1D_test.jl")

using Test
C_final, qx_final = diffusion_1D()

#Unit Test
@testset "Difference Function" begin 
    @test diff([1, 2, 3]) == [1,1]
    @test diff([10, 15, 25]) == [5, 10]
 end

# Reference Test
inds = sort(rand(1:length(C_final),20))
C_vals = C_final[inds]; qx_vals = qx_final[inds]

@testset "1D Diffusion" begin
    @test all(C_vals  .≈ C_final[inds])
    @test all(qx_vals .≈ qx_final[inds])
end 