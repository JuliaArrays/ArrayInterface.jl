# ArrayInterface.jl


[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliaarrays.github.io/ArrayInterface.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliaarrays.github.io/ArrayInterface.jl/dev)
[![CI](https://github.com/SciML/ArrayInterface.jl/workflows/CI/badge.svg)](https://github.com/SciML/ArrayInterface.jl/actions?query=workflow%3ACI)
[![CI (Julia nightly)](https://github.com/SciML/ArrayInterface.jl/workflows/CI%20(Julia%20nightly)/badge.svg)](https://github.com/SciML/ArrayInterface.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22)
[![Build status](https://badge.buildkite.com/a2db252d92478e1d7196ee7454004efdfb6ab59496cbac91a2.svg?branch=master)](https://buildkite.com/julialang/arrayinterface-dot-jl)
[![codecov](https://codecov.io/gh/SciML/ArrayInterface.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/ArrayInterface.jl)

The AbstractArray interface in Base Julia is still relatively young.
The purpose of this library is to solidify extensions to the current
AbstractArray interface, which are put to use in package ecosystems like
DifferentialEquations.jl. Since these libraries are live, this package will
serve as a staging ground for ideas before they are merged into Base Julia. For this
reason, no functionality is exported so that if such functions are added
and exported in a future Base Julia, there will be no issues with the upgrade.

# List of things to add

- https://github.com/JuliaLang/julia/issues/22216
- https://github.com/JuliaLang/julia/issues/22218
- https://github.com/JuliaLang/julia/issues/22622
- https://github.com/JuliaLang/julia/issues/25760
- https://github.com/JuliaLang/julia/issues/25107

# Array Types to Handle

The following common array types are being understood and tested as part of this
development.

- Array
- Various versions of sparse arrays
- SArray
- MArray
- FieldVector
- ArrayPartition
- VectorOfArray
- DistributedArrays
- GPUArrays (CLArrays and CuArrays)
- AFArrays
- MultiScaleArrays
- LabelledArrays

## Breaking Release Notes

2.0: Changed the default of `ismutable(array::AbstractArray) = true`. We previously defaulted to
`Base.@pure ismutable(array::AbstractArray) = typeof(array).mutable`, but there are a lot of cases
where this tends to not work out in a way one would expect. For example, if you put a normal array
into an immutable struct that adds more information to it, this is considered immutable, even if
all of the `setindex!` methods work (by forwarding to the mutable array). Thus, it seems safer to just
always assume mutability is standard for an array, and allow arrays to opt-out.
