module ArrayInterface

using ArrayInterfaceCore
import ArrayInterfaceCore: allowed_getindex, allowed_setindex!, aos_to_soa, buffer,
    has_parent, parent_type, fast_matrix_colors, findstructralnz, has_sparsestruct,
    issingular, is_lazy_conjugate, isstructured, matrix_colors, restructure, lu_instance,
    safevec, unsafe_reconstruct, zeromatrix

# ArrayIndex subtypes and methods
import ArrayInterfaceCore: ArrayIndex, MatrixIndex, VectorIndex, BidiagonalIndex, TridiagonalIndex
# managing immutables
import ArrayInterfaceCore: ismutable, can_change_size, can_setindex, deleteat, insert
# constants
import ArrayInterfaceCore: MatAdjTrans, VecAdjTrans, UpTri, LoTri
# device pieces
import ArrayInterfaceCore: AbstractDevice, AbstractCPU, CPUPointer, CPUTuple, CheckParent,
    CPUIndex, GPU, can_avx

using Static
using Static: Zero, One, nstatic, eq, ne, gt, ge, lt, le, eachop, eachop_tuple,
    permute, invariant_permutation, field_type, reduce_tup

using IfElse

using Base.Cartesian
using Base: @propagate_inbounds, tail, OneTo, LogicalIndex, Slice, ReinterpretArray,
    ReshapedArray, AbstractCartesianIndex

using Base.Iterators: Pairs
using LinearAlgebra

import Compat

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

@inline function _to_cartesian(a, i::CanonicalInt)
    @inbounds(CartesianIndices(ntuple(dim -> indices(a, dim), Val(ndims(a))))[i])
end
@inline function _to_linear(a, i::Tuple{CanonicalInt,Vararg{CanonicalInt}})
    _strides2int(offsets(a), size_to_strides(size(a), static(1)), i) + static(1)
end

"""
    device(::Type{T}) -> AbstractDevice

Indicates the most efficient way to access elements from the collection in low-level code.
For `GPUArrays`, will return `ArrayInterfaceCore.GPU()`.
For `AbstractArray` supporting a `pointer` method, returns `ArrayInterfaceCore.CPUPointer()`.
For other `AbstractArray`s and `Tuple`s, returns `ArrayInterfaceCore.CPUIndex()`.
Otherwise, returns `nothing`.
"""
device(A) = device(typeof(A))
device(::Type) = nothing
device(::Type{<:Tuple}) = CPUTuple()
device(::Type{T}) where {T<:Array} = CPUPointer()
device(::Type{T}) where {T<:AbstractArray} = _device(has_parent(T), T)
function _device(::True, ::Type{T}) where {T}
    if defines_strides(T)
        return device(parent_type(T))
    else
        return _not_pointer(device(parent_type(T)))
    end
end
_not_pointer(::CPUPointer) = CPUIndex()
_not_pointer(x) = x
_device(::False, ::Type{T}) where {T<:DenseArray} = CPUPointer()
_device(::False, ::Type{T}) where {T} = CPUIndex()

include("array_index.jl")
include("ranges.jl")
include("axes.jl")
include("size.jl")
include("dimensions.jl")
include("indexing.jl")
include("stridelayout.jl")
include("broadcast.jl")

end
