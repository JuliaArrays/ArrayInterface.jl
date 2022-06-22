module ArrayInterfaceStaticArrays

using Adapt
using ArrayInterface
using LinearAlgebra
using StaticArrays
using Static

const CanonicalInt = Union{Int,StaticInt}

ArrayInterface.ismutable(::Type{<:StaticArrays.StaticArray}) = false
ArrayInterface.ismutable(::Type{<:StaticArrays.MArray}) = true
ArrayInterface.ismutable(::Type{<:StaticArrays.SizedArray}) = true

ArrayInterface.can_setindex(::Type{<:StaticArrays.StaticArray}) = false
ArrayInterface.buffer(A::Union{StaticArrays.SArray,StaticArrays.MArray}) = getfield(A, :data)

function ArrayInterface.lu_instance(_A::StaticArrays.StaticMatrix{N,N}) where {N}
    A = StaticArrays.SArray(_A)
    L = LowerTriangular(A)
    U = UpperTriangular(A)
    p = StaticArrays.SVector{N,Int}(1:N)
    return StaticArrays.LU(L, U, p)
end

function ArrayInterface.restructure(x::StaticArrays.SArray, y::StaticArrays.SArray)
    reshape(y, StaticArrays.Size(x))
end
ArrayInterface.restructure(x::StaticArrays.SArray{S}, y) where {S} = StaticArrays.SArray{S}(y)

ArrayInterface.known_first(::Type{<:StaticArrays.SOneTo}) = 1
ArrayInterface.known_last(::Type{StaticArrays.SOneTo{N}}) where {N} = N
ArrayInterface.known_length(::Type{StaticArrays.SOneTo{N}}) where {N} = N
ArrayInterface.known_length(::Type{StaticArrays.Length{L}}) where {L} = L
function ArrayInterface.known_length(::Type{A}) where {A<:StaticArrays.StaticArray}
    ArrayInterface.known_length(StaticArrays.Length(A))
end

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
@generated function ArrayInterface.size(A::StaticArrays.StaticArray{S}) where {S}
    t = Expr(:tuple)
    Sp = S.parameters
    for n = 1:length(Sp)
        push!(t.args, Expr(:call, Expr(:curly, :StaticInt, Sp[n])))
    end
    return t
end
@generated function ArrayInterface.strides(A::StaticArrays.StaticArray{S}) where {S}
    t = Expr(:tuple, Expr(:call, Expr(:curly, :StaticInt, 1)))
    Sp = S.parameters
    x = 1
    for n = 1:(length(Sp)-1)
        push!(t.args, Expr(:call, Expr(:curly, :StaticInt, (x *= Sp[n]))))
    end
    return t
end
if StaticArrays.SizedArray{Tuple{8,8},Float64,2,2} isa UnionAll
    @inline ArrayInterface.strides(B::StaticArrays.SizedArray{S,T,M,N,A}) where {S,T,M,N,A<:SubArray} = ArrayInterface.strides(B.data)
    ArrayInterface.parent_type(::Type{<:StaticArrays.SizedArray{S,T,M,N,A}}) where {S,T,M,N,A} = A
else
    ArrayInterface.parent_type(::Type{<:StaticArrays.SizedArray{S,T,M,N}}) where {S,T,M,N} = Array{T,N}
end

Adapt.adapt_storage(::Type{<:StaticArrays.SArray{S}}, xs::Array) where {S} = SArray{S}(xs)

end # module
