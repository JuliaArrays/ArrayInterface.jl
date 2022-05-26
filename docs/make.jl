using ArrayInterface
using Pkg

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.develop(PackageSpec(path=subpkg_path))
end
dev_subpkg("ArrayInterfaceCore")

using ArrayInterfaceCore
using Documenter

makedocs(;
    modules=[ArrayInterface, ArrayInterfaceCore],
    sitename="ArrayInterface",
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ]
)

deploydocs(;
    repo="github.com/JuliaArrays/ArrayInterface.jl"
)
