module ArrayInterfaceStaticArraysExt

using Adapt
using ArrayInterface
using LinearAlgebra
if isdefined(Base, :get_extension) 
    using StaticArrays
    using Static
    using Static: StaticInt
else 
    using ..StaticArrays
    using ..Static
    using ..Static: StaticInt
end

const CanonicalInt = Union{Int,StaticInt}

function Static.OptionallyStaticUnitRange(::StaticArrays.SOneTo{N}) where {N}
    Static.OptionallyStaticUnitRange(StaticInt(1), StaticInt(N))
end
ArrayInterface.known_first(::Type{<:StaticArrays.SOneTo}) = 1
ArrayInterface.known_last(::Type{StaticArrays.SOneTo{N}}) where {N} = N
ArrayInterface.known_length(::Type{StaticArrays.SOneTo{N}}) where {N} = N
ArrayInterface.known_length(::Type{StaticArrays.Length{L}}) where {L} = L
function ArrayInterface.known_length(::Type{A}) where {A<:StaticArrays.StaticArray}
    ArrayInterface.known_length(StaticArrays.Length(A))
end

@inline ArrayInterface.static_length(x::StaticArrays.StaticArray) = Static.maybe_static(ArrayInterface.known_length, Base.length, x)
ArrayInterface.device(::Type{<:StaticArrays.MArray}) = ArrayInterface.CPUPointer()
ArrayInterface.device(::Type{<:StaticArrays.SArray}) = ArrayInterface.CPUTuple()
ArrayInterface.contiguous_axis(::Type{<:StaticArrays.StaticArray}) = StaticInt{1}()
ArrayInterface.contiguous_batch_size(::Type{<:StaticArrays.StaticArray}) = StaticInt{0}()
function ArrayInterface.stride_rank(::Type{T}) where {N,T<:StaticArray{<:Any,<:Any,N}}
    ntuple(static, StaticInt(N))
end
function ArrayInterface.dense_dims(::Type{<:StaticArray{S,T,N}}) where {S,T,N}
    ArrayInterface._all_dense(Val(N))
end
ArrayInterface.defines_strides(::Type{<:StaticArrays.SArray}) = true
ArrayInterface.defines_strides(::Type{<:StaticArrays.MArray}) = true

@generated function ArrayInterface.axes_types(::Type{<:StaticArrays.StaticArray{S}}) where {S}
    Tuple{[StaticArrays.SOneTo{s} for s in S.parameters]...}
end
@generated function ArrayInterface.static_size(A::StaticArrays.StaticArray{S}) where {S}
    t = Expr(:tuple)
    Sp = S.parameters
    for n = 1:length(Sp)
        push!(t.args, Expr(:call, Expr(:curly, :StaticInt, Sp[n])))
    end
    return t
end
@generated function ArrayInterface.static_strides(A::StaticArrays.StaticArray{S}) where {S}
    t = Expr(:tuple, Expr(:call, Expr(:curly, :StaticInt, 1)))
    Sp = S.parameters
    x = 1
    for n = 1:(length(Sp)-1)
        push!(t.args, Expr(:call, Expr(:curly, :StaticInt, (x *= Sp[n]))))
    end
    return t
end
if StaticArrays.SizedArray{Tuple{8,8},Float64,2,2} isa UnionAll
    @inline ArrayInterface.static_strides(B::StaticArrays.SizedArray{S,T,M,N,A}) where {S,T,M,N,A<:SubArray} = ArrayInterface.static_strides(B.data)
    ArrayInterface.parent_type(::Type{<:StaticArrays.SizedArray{S,T,M,N,A}}) where {S,T,M,N,A} = A
else
    ArrayInterface.parent_type(::Type{<:StaticArrays.SizedArray{S,T,M,N}}) where {S,T,M,N} = Array{T,N}
end

end # module
