import Pkg

const GROUP = get(ENV, "GROUP", "All")

function dev_subpkg(subpkg)
    subpkg_path = joinpath("lib", subpkg)
    Pkg.develop(Pkg.PackageSpec(path=subpkg_path))
end

function activate_subpkg_env(subpkg)
    subpkg_path = joinpath("lib", subpkg)
    Pkg.activate(subpkg_path)
    Pkg.develop(Pkg.PackageSpec(path=subpkg_path))
    Pkg.instantiate()
end

Pkg.update()

# All packages need the core
dev_subpkg("ArrayInterfaceCore")
Pkg.develop("ArrayInterface")

Pkg.test("ArrayInterface"; coverage=true)