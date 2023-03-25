```@meta
CurrentModule = ArrayInterface
```
# ArrayInterface

Designs for new Base array interface primitives, used widely through scientific machine learning (SciML) and other organizations

## Extensions

ArrayInterface.jl uses extension packages in order to add support for popular libraries to its interface functions. These packages are:

- BandedMatrices.jl
- BlockBandedMatrices.jl
- GPUArrays.jl / CUDA.jl
- OffsetArrays.jl
- Tracker.jl

## StaticArrayInterface.jl

If one is looking for an interface which includes functionality for statically-computed values, see
[StaticArrayInterface.jl](https://github.com/JuliaArrays/StaticArrayInterface.jl).
This was separated from ArrayInterface.jl because it includes a lot of functionality that does not give substantive improvements
to the interface, and is likely to be deprecated in the near future as the compiler matures to automate a lot of its optimizations.