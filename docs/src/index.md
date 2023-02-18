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

## API

### Traits

```@docs
ArrayInterfaceCore.can_avx
ArrayInterfaceCore.can_change_size
ArrayInterfaceCore.can_setindex
ArrayInterfaceCore.device
ArrayInterfaceCore.defines_strides
ArrayInterfaceCore.fast_matrix_colors
ArrayInterfaceCore.fast_scalar_indexing
ArrayInterfaceCore.indices_do_not_alias
ArrayInterfaceCore.instances_do_not_alias
ArrayInterfaceCore.is_forwarding_wrapper
ArrayInterfaceCore.ismutable
ArrayInterfaceCore.isstructured
ArrayInterfaceCore.has_sparsestruct
ArrayInterfaceCore.ndims_index
ArrayInterfaceCore.ndims_shape

```

### Functions

```@docs
ArrayInterfaceCore.allowed_getindex
ArrayInterfaceCore.allowed_setindex!
ArrayInterfaceCore.aos_to_soa
ArrayInterfaceCore.buffer
ArrayInterfaceCore.findstructralnz
ArrayInterfaceCore.lu_instance
ArrayInterfaceCore.matrix_colors
ArrayInterfaceCore.issingular
ArrayInterfaceCore.parent_type
ArrayInterfaceCore.promote_eltype
ArrayInterfaceCore.restructure
ArrayInterfaceCore.safevec
ArrayInterfaceCore.zeromatrix
ArrayInterfaceCore.undefmatrix
```

### Types

```@docs
ArrayInterfaceCore.ArrayIndex
ArrayInterfaceCore.GetIndex
ArrayInterfaceCore.SetIndex!
```