module ArrayInterface

using ArrayInterfaceCore
import ArrayInterfaceCore: allowed_getindex, allowed_setindex!, aos_to_soa, buffer,
    has_parent, parent_type, fast_matrix_colors, findstructralnz, has_sparsestruct,
    issingular, is_lazy_conjugate, isstructured, matrix_colors, restructure, lu_instance,
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
    known_strides, known_offsets, offsets, offset1, known_offset1, contiguous_axis, dense_dims,
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

const CanonicalInt = Union{Int,StaticInt}
canonicalize(x::Integer) = Int(x)
canonicalize(@nospecialize(x::StaticInt)) = x

abstract type AbstractArray2{T,N} <: AbstractArray{T,N} end

Base.size(A::AbstractArray2) = map(Int, ArrayInterfaceCore.size(A))
Base.size(A::AbstractArray2, dim) = Int(ArrayInterfaceCore.size(A, dim))

function Base.axes(A::AbstractArray2)
    !(parent_type(A) <: typeof(A)) && return ArrayInterfaceCore.axes(parent(A))
    throw(ArgumentError("Subtypes of `AbstractArray2` must define an axes method"))
end
Base.axes(A::AbstractArray2, dim) = ArrayInterfaceCore.axes(A, dim)

function Base.strides(A::AbstractArray2)
    defines_strides(A) && return map(Int, ArrayInterfaceCore.strides(A))
    throw(MethodError(Base.strides, (A,)))
end
Base.strides(A::AbstractArray2, dim) = Int(ArrayInterfaceCore.strides(A, dim))

function Base.IndexStyle(::Type{T}) where {T<:AbstractArray2}
    if parent_type(T) <: T
        return IndexCartesian()
    else
        return IndexStyle(parent_type(T))
    end
end

function Base.length(A::AbstractArray2)
    len = known_length(A)
    if len === nothing
        return Int(prod(size(A)))
    else
        return Int(len)
    end
end

@propagate_inbounds Base.getindex(A::AbstractArray2, args...) = getindex(A, args...)
@propagate_inbounds Base.getindex(A::AbstractArray2; kwargs...) = getindex(A; kwargs...)

@propagate_inbounds function Base.setindex!(A::AbstractArray2, val, args...)
    return setindex!(A, val, args...)
end
@propagate_inbounds function Base.setindex!(A::AbstractArray2, val; kwargs...)
    return setindex!(A, val; kwargs...)
end

@inline static_first(x) = Static.maybe_static(known_first, first, x)
@inline static_last(x) = Static.maybe_static(known_last, last, x)
@inline static_step(x) = Static.maybe_static(known_step, step, x)

include("array_index.jl")
include("axes.jl")
include("broadcast.jl")
include("dimensions.jl")
include("indexing.jl")
include("ranges.jl")
include("size.jl")
include("stridelayout.jl")

end
