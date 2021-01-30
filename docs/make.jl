

using Documenter
using ArrayInterface

makedocs(;
    modules=[ArrayInterface],
    format=Documenter.HTML(),
    pages=[
        "ArrayInterface" => "index.md",
        "Examples" => "examples.md",
        "API" => "api.md"
   ],
    repo="https://github.com/SciML/ArrayInterface.jl/blob/{commit}{path}#L{line}",
    sitename="ArrayInterface.jl",
)

deploydocs(repo = "github.com/SciML/ArrayInterface.jl.git")

