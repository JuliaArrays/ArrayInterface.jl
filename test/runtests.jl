using SafeTestsets, Pkg

const GROUP = get(ENV, "GROUP", "All")

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "BandedMatrices" begin include("bandedmatrices.jl") end
        @time @safetestset "BlockBandedMatrices" begin include("blockbandedmatrices.jl") end
        @time @safetestset "Core" begin include("core.jl") end
        @time @safetestset "StaticArraysCore" begin include("staticarrayscore.jl") end
    end

    if GROUP == "GPU"
        activate_gpu_env()
        @time @safetestset "CUDA" begin include("gpu/cuda.jl") end
    end
end