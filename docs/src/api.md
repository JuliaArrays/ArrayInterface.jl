# API

## ArrayInterfaceCore.jl

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
ArrayInterfaceCore.restructure
ArrayInterfaceCore.safevec
ArrayInterfaceCore.zeromatrix
```

### Types

```@docs
ArrayInterfaceCore.ArrayIndex
ArrayInterfaceCore.GetIndex
ArrayInterfaceCore.SetIndex!
```

## ArrayInterface.jl

### Traits

```@docs
ArrayInterface.contiguous_axis
ArrayInterface.contiguous_axis_indicator
ArrayInterface.contiguous_batch_size
ArrayInterface.dimnames
ArrayInterface.has_dimnames
ArrayInterface.has_parent
ArrayInterface.is_column_major
ArrayInterface.is_lazy_conjugate
ArrayInterface.is_splat_index
ArrayInterface.known_dimnames
ArrayInterface.known_first
ArrayInterface.known_last
ArrayInterface.known_length
ArrayInterface.known_offset1
ArrayInterface.known_offsets
ArrayInterface.known_size
ArrayInterface.known_step
ArrayInterface.known_strides
```

### Functions

```@docs
ArrayInterface.axes
ArrayInterface.axes_types
ArrayInterface.broadcast_axis
ArrayInterface.deleteat
ArrayInterface.dense_dims
ArrayInterface.from_parent_dims
ArrayInterface.getindex
ArrayInterface.indices
ArrayInterface.insert
ArrayInterface.length
ArrayInterface.lazy_axes
ArrayInterface.offset1
ArrayInterface.offsets
ArrayInterface.setindex!
ArrayInterface.size
ArrayInterface.strides
ArrayInterface.to_axes
ArrayInterface.to_axis
ArrayInterface.to_dims
ArrayInterface.to_index
ArrayInterface.to_indices
ArrayInterface.to_parent_dims
ArrayInterface.unsafe_reconstruct
```

### Types

```@docs
ArrayInterface.BroadcastAxis
ArrayInterface.LazyAxis
ArrayInterface.OptionallyStaticStepRange
ArrayInterface.OptionallyStaticUnitRange
ArrayInterface.SOneTo
ArrayInterface.SUnitRange
ArrayInterface.StrideIndex
```

