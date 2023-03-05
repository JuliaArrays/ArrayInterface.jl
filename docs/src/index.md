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
ArrayInterface.can_avx
ArrayInterface.can_change_size
ArrayInterface.can_setindex
ArrayInterface.device
ArrayInterface.defines_strides
ArrayInterface.ensures_all_unique
ArrayInterface.ensures_sorted
ArrayInterface.fast_matrix_colors
ArrayInterface.fast_scalar_indexing
ArrayInterface.indices_do_not_alias
ArrayInterface.instances_do_not_alias
ArrayInterface.is_forwarding_wrapper
ArrayInterface.ismutable
ArrayInterface.isstructured
ArrayInterface.has_sparsestruct
ArrayInterface.ndims_index
ArrayInterface.ndims_shape

```

### Functions

```@docs
ArrayInterface.allowed_getindex
ArrayInterface.allowed_setindex!
ArrayInterface.aos_to_soa
ArrayInterface.buffer
ArrayInterface.findstructralnz
ArrayInterface.flatten_tuples
ArrayInterface.lu_instance
ArrayInterface.map_tuple_type
ArrayInterface.matrix_colors
ArrayInterface.issingular
ArrayInterface.parent_type
ArrayInterface.promote_eltype
ArrayInterface.restructure
ArrayInterface.safevec
ArrayInterface.zeromatrix
ArrayInterface.undefmatrix
```

### Types

```@docs
ArrayInterface.ArrayIndex
ArrayInterface.GetIndex
ArrayInterface.SetIndex!
```