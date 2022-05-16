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
dev_subpkg("ArrayInterfaceCore")
if GROUP == "ArrayInterfaceBlockBandedMatrices"
    dev_subpkg("ArrayInterfaceBandedMatrices")
end

@time begin
if GROUP == "HighLevel"
    # Any tests for the level combined package? None right now.
else
    dev_subpkg(GROUP)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", GROUP)
    Pkg.test(PackageSpec(name=GROUP, path=subpkg_path))
end
end