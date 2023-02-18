using ArrayInterface
using Documenter

makedocs(;
    modules=[ArrayInterface],
    sitename="ArrayInterface.jl",
    pages=[
        "ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming" => "index.md",
    ]
)

deploydocs(;
    repo="github.com/JuliaArrays/ArrayInterface.jl"
)
