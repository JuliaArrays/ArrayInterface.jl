# API

## ArrayInterfaceCore.jl

### Traits

```@docs
ArrayInterfaceCore.can_avx
ArrayInterfaceCore.can_change_size
ArrayInterfaceCore.can_setindex
ArrayInterfaceCore.fast_matrix_colors
ArrayInterfaceCore.fast_scalar_indexing
ArrayInterfaceCore.ismutable
ArrayInterfaceCore.isstructured
ArrayInterfaceCore.has_sparsestruct
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
```

## ArrayInterface.jl

### Traits

```@docs
ArrayInterface.contiguous_axis
ArrayInterface.contiguous_axis_indicator
ArrayInterface.contiguous_batch_size
ArrayInterface.defines_strides
ArrayInterface.device
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
ArrayInterface.ndims_index
```

### Functions

```@docs
ArrayInterface.axes
ArrayInterface.axes_types
ArrayInterface.broadcast_axis
ArrayInterface.deleteat
ArrayInterface.dense_dims
ArrayInterface.getindex
ArrayInterface.indices
ArrayInterface.insert
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
ArrayInterface.unsafe_reconstruct
```

### Types

```@docs
ArrayInterface.BroadcastAxis
ArrayInterface.LazyAxis
ArrayInterface.OptionallyStaticStepRange
ArrayInterface.OptionallyStaticUnitRange
ArrayInteraface.SOneTo
ArrayInteraface.SUnitRange
ArrayInterface.StrideIndex
```

