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
        @time @safetestset "StaticArrays" begin include("staticarrays.jl") end
        @time @safetestset "StaticArraysCore" begin include("staticarrayscore.jl") end

        @time @safetestset "Static" begin
            include("static/setup.jl")
            include("static/array_index.jl")
            include("static/axes.jl")
            include("static/broadcast.jl")
            include("static/dimensions.jl")
            include("static/indexing.jl")
            include("static/ranges.jl")
            include("static/size.jl")
            include("static/stridelayout.jl")
            include("static/misc.jl")
        end
        @time @safetestset "OffsetArrays" begin include("offsetarrays.jl") end
    end

    if GROUP == "GPU"
        activate_gpu_env()
        @time @safetestset "CUDA" begin include("gpu/cuda.jl") end
    end
end

