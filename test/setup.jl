using ArrayInterface, Static, LinearAlgebra

struct MArray{T,N,R} <: DenseArray{T,N}
    parent::Array{T,N}
    indices::LinearIndices{N,R}
end

MArray(A::Array) = MArray(A, LinearIndices(map(s -> static(1):static(s), size(A))))
Base.parent(x::MArray) = x.parent
Base.IndexStyle(::Type{<:MArray}) = IndexLinear()
ArrayInterface.axes(x::MArray) = ArrayInterface.axes(x.indices)
Base.axes(x::MArray) = ArrayInterface.axes(x)
ArrayInterface.axes_types(T::Type{<:MArray}) = T.parameters[3]
#ArrayInterface.size(x::MArray) = ArrayInterface.size(x.indices)
ArrayInterface.defines_strides(::Type{<:MArray}) = true
Base.strides(x::MArray) = strides(parent(x))
function Base.getindex(x::MArray, inds...)
    @boundscheck checkbounds(x, inds...)
    @inbounds parent(x)[inds...]
end

Base.size(x::MArray) = map(Int, ArrayInterface.size(x))

struct NamedDimsWrapper{D,T,N,P<:AbstractArray{T,N}} <: ArrayInterface.AbstractArray2{T,N}
    dimnames::D
    parent::P
    NamedDimsWrapper(d::D, p::P) where {D,P} = new{D,eltype(P),ndims(p),P}(d, p)
end
ArrayInterface.is_forwarding_wrapper(::Type{<:NamedDimsWrapper}) = true
Base.parent(x::NamedDimsWrapper) = getfield(x, :parent)
ArrayInterface.parent_type(::Type{T}) where {P,T<:NamedDimsWrapper{<:Any,<:Any,<:Any,P}} = P
ArrayInterface.dimnames(x::NamedDimsWrapper) = getfield(x, :dimnames)
function ArrayInterface.known_dimnames(::Type{T}) where {L,T<:NamedDimsWrapper{L}}
    ArrayInterface.Static.known(L)
end

struct LabelledArray{T,N,P<:AbstractArray{T,N},L} <: ArrayInterface.AbstractArray2{T,N}
    parent::P
    labels::L

    LabelledArray(p::P, labels::L) where {P,L} = new{eltype(P),ndims(p),P,L}(p, labels)
end
ArrayInterface.is_forwarding_wrapper(::Type{<:LabelledArray}) = true
Base.parent(x::LabelledArray) = getfield(x, :parent)
ArrayInterface.parent_type(::Type{T}) where {P,T<:LabelledArray{<:Any,<:Any,P}} = P
ArrayInterface.index_labels(x::LabelledArray) = map(ArrayInterface.LabelledIndices, getfield(x, :labels))

# Dummy array type with undetermined contiguity properties
struct DummyZeros{T,N} <: AbstractArray{T,N}
    dims :: Dims{N}
    DummyZeros{T}(dims...) where {T} = new{T,length(dims)}(dims)
end
DummyZeros(dims...) = DummyZeros{Float64}(dims...)
Base.size(x::DummyZeros) = x.dims
Base.getindex(::DummyZeros{T}, inds...) where {T} = zero(T)

struct Wrapper{T,N,P<:AbstractArray{T,N}} <: ArrayInterface.AbstractArray2{T,N}
    parent::P
end
ArrayInterface.parent_type(::Type{<:Wrapper{T,N,P}}) where {T,N,P} = P
Base.parent(x::Wrapper) = x.parent
ArrayInterface.is_forwarding_wrapper(::Type{<:Wrapper}) = true

struct DenseWrapper{T,N,P<:AbstractArray{T,N}} <: DenseArray{T,N} end
ArrayInterface.parent_type(::Type{DenseWrapper{T,N,P}}) where {T,N,P} = P
