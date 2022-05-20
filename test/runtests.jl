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

@show GROUP, GROUP == "ArrayInterface"

@time begin
if GROUP == "ArrayInterface"
    include("setup.jl")
    include("array_index.jl")
    include("axes.jl")
    include("broadcast.jl")
    include("dimensions.jl")
    include("indexing.jl")
    include("ranges.jl")
    include("setup.jl")
    include("size.jl")
else
    dev_subpkg(GROUP)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", GROUP)
    Pkg.test(PackageSpec(name=GROUP, path=subpkg_path))
end
end