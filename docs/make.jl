using ArrayInterface
using Documenter

makedocs(;
    modules=[ArrayInterface],
    sitename="ArrayInterface.jl",
    pages=[
        "ArrayInterface.jl: An Extended Array Interface for Julia Generic Programming" => "index.md",
        "indexing.md",
        "conversions.md",
        "linearalgebra.md",
        "sparsearrays.md",
        "tuples.md",
        "wrapping.md",
    ]
)

deploydocs(;
    repo="github.com/JuliaArrays/ArrayInterface.jl"
)
