
_as_index(x) = x
_as_index(x::Integer) = static(1):x
_as_index(x::OneTo) = static(1):length(x)
_as_index(x::StepRange) = OptionallyStaticStepRange(x)
_as_index(x::UnitRange) = OptionallyStaticUnitRange(x)
_as_index(x::OptionallyStaticRange) = x

"""
    StrideLayout(A)

Produces an array whose elements correspond to the linear buffer position of `A`'s elements.
"""
struct StrideLayout{N,R,O1,S,A<:Tuple{Vararg{Any,N}}} <: AbstractArray2{Int,N}
    rank::R
    offset1::O1
    strides::S
    axes::A
end

offset1(x::StrideLayout) = getfield(x, :offset1)
offsets(x::StrideLayout) = map(static_first, axes(x))
axes(x::StrideLayout) = getfield(x, :axes)
@inline function axes(x::StrideLayout{N}, i::Int) where {N}
    if i > N
        return static(1):1
    else
        return getfield(getfield(x, :axes), i)
    end
end
@inline function axes(x::StrideLayout{N}, ::StaticInt{i}) where {N,i}
    if i > N
        return static(1):static(1)
    else
        return getfield(getfield(x, :axes), i)
    end
end
strides(x::StrideLayout) = getfield(x, :strides)
stride_rank(x::StrideLayout) = getfield(x, :rank)

@inline function StrideLayout(x::DenseArray)
    a = axes(x)
    return StrideLayout(
        stride_rank(x),
        offset1(x),
        size_to_strides(map(static_length, a), static(1)),
        a
    )
end

# TODO optimize this
@inline function StrideLayout(x)
    return StrideLayout(
        stride_rank(x),
        offset1(x),
        strides(x),
        axes(x)
    )
end

##############
### layout ###
##############
layout(x, i) = layout(x)
layout(x, i::AbstractVector{<:Integer}) = _maybe_linear_layout(IndexStyle(x), x)
layout(x, i::Integer) = _maybe_linear_layout(IndexStyle(x), x)
layout(x, i::AbstractCartesianIndex{1}) = _maybe_linear_layout(IndexStyle(x), x)
function layout(x, i::AbstractVector{AbstractCartesianIndex{1}})
    return _maybe_linear_layout(IndexStyle(x), x)
end
_maybe_linear_layout(::IndexLinear, x) = _as_index(eachindex(x))
_maybe_linear_layout(::IndexStyle, x) = layout(x)
layout(x::StrideLayout) = x
layout(x::LinearIndices) = x
layout(x::CartesianIndices) = x
function layout(x)
    if defines_strides(x)
        return StrideLayout(x)
    else
        return _layout_indices(IndexStyle(x), axes(x))
    end
end
function layout(x::Transpose)
    if defines_strides(x)
        return StrideLayout(x)
    else
        return Transpose(layout(parent(x)))
    end
end
function layout(x::Adjoint{T}) where {T<:Number}
    if defines_strides(x)
        return StrideLayout(x)
    else
        return Transpose(layout(parent(x)))
    end
end
function layout(x::PermutedDimsArray{T,N,perm,iperm}) where {T,N,perm,iperm}
    if defines_strides(x)
        return StrideLayout(x)
    else
        p = layout(parent(x))
        return PermutedDimsArray{eltype(p),ndims(p),perm, iperm,typeof(p)}(p)
    end
end
function layout(x::SubArray)
    if defines_strides(x)
        return StrideLayout(x)
    else
        return @inbounds(view(layout(parent(x)), x.indices...))
    end
end
_layout_indices(::IndexStyle, axs) = CartesianIndices(axs)
_layout_indices(::IndexLinear, axs) = LinearIndices(axs)

"""
    buffer(x)

Return the raw buffer for `x`, stripping any additional info (structural, indexing,
metadata, etc.).
"""
buffer(x) = x
@inline buffer(x::PermutedDimsArray) = buffer(parent(x))
@inline buffer(x::Transpose) = buffer(parent(x))
@inline buffer(x::Adjoint) = buffer(parent(x))
@inline buffer(x::SubArray) = buffer(parent(x))


""" allocate_memory(::AbstractDevice, ::Type{T}, length::Union{StaticInt,Int}) """
allocate_memory(::CPUPointer, ::Type{T}, ::StaticInt{N}) where {T,N} = Ref{NTuple{N,T}}
allocate_memory(::CPUPointer, ::Type{T}, n::Int) where {T} = Vector{T}(undef, n)
allocate_memory(::CPUTuple, ::Type{T}, ::StaticInt{N}) where {T,N} = Ref{NTuple{N,T}}


""" dereference(::AbstractDevice, x) """
dereference(::CPUPointer, x) = x
dereference(::CPUTuple, x::Ref) = x[]

""" initialize(data, layout) """
function initialize end

