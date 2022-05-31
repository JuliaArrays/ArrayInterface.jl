using Test, Pkg

const GROUP = get(ENV, "GROUP", "All")

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.develop(PackageSpec(path=subpkg_path))
end

function activate_subpkg_env(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.activate(subpkg_path)
    Pkg.develop(PackageSpec(path=subpkg_path))
    Pkg.instantiate()
end

# Add necessary sub-dependencies
if GROUP == "ArrayInterfaceBlockBandedMatrices"
    dev_subpkg("ArrayInterfaceBandedMatrices")
end

groups = if GROUP == "All"
    ["ArrayInterfaceCore", "ArrayInterface", "ArrayInterfaceBandedMatrices", "ArrayInterfaceBlockBandedMatrices",
     "ArrayInterfaceOffsetArrays", "ArrayInterfaceStaticArrays",]
else
    [GROUP]
end

@time begin

for g in groups
    if g == "ArrayInterface"

        println("ArrayInterface Tests")

        include("setup.jl")
        include("array_index.jl")
        include("axes.jl")
        include("broadcast.jl")
        include("dimensions.jl")
        include("indexing.jl")
        include("ranges.jl")
        include("size.jl")
        include("misc.jl")
    else
        dev_subpkg(g)
        subpkg_path = joinpath(dirname(@__DIR__), "lib", g)
        Pkg.test(PackageSpec(name=g, path=subpkg_path))
    end
end

end
