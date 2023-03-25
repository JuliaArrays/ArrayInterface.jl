# Julia's Extended Array Indexing Interface

The following ArrayInterface functions extend Julia's Base LinearAlgebra interface
to improve the ability to write code for generic array types.

## Indexing Traits

The following traits allow for one to accurately determine the type of indexing allowed
on arrays in order to write optimal code for generic array types.

```@docs
ArrayInterface.can_avx
ArrayInterface.can_change_size
ArrayInterface.can_setindex
ArrayInterface.fast_scalar_indexing
ArrayInterface.ismutable
ArrayInterface.ndims_index
ArrayInterface.ndims_shape
ArrayInterface.defines_strides
ArrayInterface.ensures_all_unique
ArrayInterface.ensures_sorted
ArrayInterface.indices_do_not_alias
ArrayInterface.instances_do_not_alias
ArrayInterface.device
```

## Allowed Indexing Functions

These are generic functions for forced "allowed indexing". For example, with CUDA.jl's
CuArrays a mode can be enabled such that `allowscalar(false)` forces errors to be thrown
if a GPU array is scalar indexed. Instead of using the CUDA-specific `CUDA.@allowscalar`
on an operation, these functions allow for a general generic "allowed indexing" for all
array types.

```@docs
ArrayInterface.allowed_getindex
ArrayInterface.allowed_setindex!
```

## Indexing Type Buffers

The following indexing types allow for generically controlling bounds checking
and index translations.

```@docs
ArrayInterface.ArrayIndex
ArrayInterface.GetIndex
ArrayInterface.SetIndex!
```