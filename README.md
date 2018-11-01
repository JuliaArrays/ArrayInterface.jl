[![Build Status](https://travis-ci.org/JuliaDiffEq/ArrayInterface.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/ArrayInterface.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/s4vnsj386dyyv655?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/arrayinterface-jl)

# ArrayInterface.jl

Julia has only recently reached v1.0 and the AbstractArray interface is still
quite new. The purpose of this library is to solidify extensions to the current
AbstractArray interface which are put to use in package ecosystems like
DifferentialEquations.jl. Since these libraries are live, this package will
serve as a staging ground for ideas before they merged into Base Julia. For this
reason, no functionality is exported so that way if such functions are added
and exported in a future Base Julia there will be no issues with the upgrade.

## ismutable(x)

A trait function for whether `x` is a mutable or immutable array. Used for
dispatching to in-place and out-of-place versions of functions.

## List of things to add

- https://github.com/JuliaLang/julia/issues/22216
- https://github.com/JuliaLang/julia/issues/22218
- https://github.com/JuliaLang/julia/issues/22622
- https://github.com/JuliaLang/julia/issues/25760
- https://github.com/JuliaLang/julia/issues/25107

## Array Types to Handle

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
