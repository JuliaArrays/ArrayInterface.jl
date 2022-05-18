module ArrayInterface

using ArrayInterfaceCore
import ArrayInterfaceCore: allowed_getindex, allowed_setindex!, aos_to_soa, buffer,
    has_parent, parent_type, fast_matrix_colors,  findstructralnz, has_sparsestruct,
    issingular, is_lazy_conjugate,  isstructured,  matrix_colors, restructure, lu_instance,
    safevec, unsafe_reconstruct, zeromatrix

# ArrayIndex subtypes and methods
import ArrayInterfaceCore: ArrayIndex, MatrixIndex, VectorIndex, BidiagonalIndex, TridiagonalIndex, StrideIndex
# device types and methods
import ArrayInterfaceCore: AbstractDevice, AbstractCPU, CPUTuple, CPUPointer, GPU, CPUIndex, CheckParent, device
# range types and methods
import ArrayInterfaceCore: OptionallyStaticStepRange, OptionallyStaticUnitRange, SOneTo,
    SUnitRange, indices, known_first, known_last, known_step, static_first, static_last, static_step
# dimension methods
import ArrayInterfaceCore: dimnames, known_dimnames, has_dimnames, from_parent_dims, to_dims, to_parent_dims
# indexing methods
import ArrayInterfaceCore: to_axes, to_axis, to_indices, to_index, getindex, setindex!,
    ndims_index, is_splat_index, fast_scalar_indexing
# stride layout methods
import ArrayInterfaceCore: strides, stride_rank, contiguous_axis_indicator, contiguous_batch_size,
    known_strides, known_offsets,offsets, offset1, known_offset1, contiguous_axis, dense_dims,
    defines_strides, is_column_major
# axes types and methods
import ArrayInterfaceCore: axes, axes_types, lazy_axes, LazyAxis
# static sizing
import ArrayInterfaceCore: size, known_size, known_length, static_length
# managing immutables
import ArrayInterfaceCore: ismutable, can_change_size, can_setindex, deleteat, insert
# constants
import ArrayInterfaceCore: MatAdjTrans, VecAdjTrans, UpTri, LoTri

using Static
using Static: Zero, One, nstatic, eq, ne, gt, ge, lt, le, eachop, eachop_tuple,
    permute, invariant_permutation, field_type, reduce_tup

end
