using ArrayInterface
using Documenter

makedocs(;
    modules=[ArrayInterface],
    sitename="ArrayInterface",
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/JuliaArrays/ArrayInterface.jl",
)

