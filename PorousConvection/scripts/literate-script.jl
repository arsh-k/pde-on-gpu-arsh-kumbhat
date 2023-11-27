using Literate

# directory where the markdown files are put
# md_dir = joinpath(@__DIR__, "md")
Literate.markdown("bin_io_script.jl", "./md", execute=true, documenter=false, credit=false)