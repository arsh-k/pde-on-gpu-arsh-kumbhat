using Test, MAT

file_cpu       = matopen("../data/ex2_task_2/2D_out_C_cpu.mat")
C_cpu          = read(file_cpu, "C"); close(file_cpu)
file_gpu       = matopen("../data/ex2_task_2/2D_out_C_gpu.mat")
C_gpu          = read(file_gpu, "C"); close(file_gpu)
file_multi_gpu = matopen("../data/2D-diffusion-implicit-global/implicitglob2D_out_C_gpu.mat")
C_multi_gpu    = read(file_multi_gpu, "C"); close(file_multi_gpu)

@testset "Single GPU & CPU Diffusion Array Comparison" begin
    @test all(C_cpu .≈ C_gpu)
end

@testset "Multi-GPU Implementation Test" begin
    @test size(C_cpu[2:end-1, 2:end-1]) == size(C_multi_gpu)
    @test size(C_gpu[2:end-1, 2:end-1]) == size(C_multi_gpu)
    @test all(C_cpu[2:end-1, 2:end-1] .≈ C_multi_gpu)
    @test all(C_gpu[2:end-1, 2:end-1] .≈ C_multi_gpu)   
end