using ArrayInterface
using Documenter

makedocs(;
    modules=[ArrayInterface],
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaArrays/ArrayInterface.jl",
)

