module ArrayInterfaceStaticArrays

using Adapt
using ArrayInterfaceCore
using LinearAlgebra
using StaticArrays
using Static

ArrayInterfaceCore.ismutable(::Type{<:StaticArrays.StaticArray}) = false
ArrayInterfaceCore.ismutable(::Type{<:StaticArrays.MArray}) = true
ArrayInterfaceCore.ismutable(::Type{<:StaticArrays.SizedArray}) = true

ArrayInterfaceCore.can_setindex(::Type{<:StaticArrays.StaticArray}) = false
ArrayInterfaceCore.buffer(A::Union{StaticArrays.SArray,StaticArrays.MArray}) = getfield(A, :data)

function ArrayInterfaceCore.lu_instance(_A::StaticArrays.StaticMatrix{N,N}) where {N}
    A = StaticArrays.SArray(_A)
    L = LowerTriangular(A)
    U = UpperTriangular(A)
    p = StaticArrays.SVector{N,Int}(1:N)
    return StaticArrays.LU(L, U, p)
end

function ArrayInterfaceCore.restructure(x::StaticArrays.SArray, y::StaticArrays.SArray)
    reshape(y, StaticArrays.Size(x))
end
ArrayInterfaceCore.restructure(x::StaticArrays.SArray{S}, y) where {S} = StaticArrays.SArray{S}(y)

ArrayInterfaceCore.known_first(::Type{<:StaticArrays.SOneTo}) = 1
ArrayInterfaceCore.known_last(::Type{StaticArrays.SOneTo{N}}) where {N} = N
ArrayInterfaceCore.known_length(::Type{StaticArrays.SOneTo{N}}) where {N} = N
ArrayInterfaceCore.known_length(::Type{StaticArrays.Length{L}}) where {L} = L
function ArrayInterfaceCore.known_length(::Type{A}) where {A <: StaticArrays.StaticArray}
    ArrayInterfaceCore.known_length(StaticArrays.Length(A))
end

ArrayInterfaceCore.device(::Type{<:StaticArrays.MArray}) = ArrayInterfaceCore.CPUPointer()
ArrayInterfaceCore.device(::Type{<:StaticArrays.SArray}) = ArrayInterfaceCore.CPUTuple()
ArrayInterfaceCore.contiguous_axis(::Type{<:StaticArrays.StaticArray}) = StaticInt{1}()
ArrayInterfaceCore.contiguous_batch_size(::Type{<:StaticArrays.StaticArray}) = StaticInt{0}()
ArrayInterfaceCore.stride_rank(::Type{T}) where {N,T<:StaticArray{<:Any,<:Any,N}} = Static.nstatic(Val(N))
function ArrayInterfaceCore.dense_dims(::Type{<:StaticArray{S,T,N}}) where {S,T,N}
    ArrayInterfaceCore._all_dense(Val(N))
end
ArrayInterfaceCore.defines_strides(::Type{<:StaticArrays.SArray}) = true
ArrayInterfaceCore.defines_strides(::Type{<:StaticArrays.MArray}) = true

@generated function ArrayInterfaceCore.axes_types(::Type{<:StaticArrays.StaticArray{S}}) where {S}
    Tuple{[StaticArrays.SOneTo{s} for s in S.parameters]...}
end
@generated function ArrayInterfaceCore.size(A::StaticArrays.StaticArray{S}) where {S}
    t = Expr(:tuple)
    Sp = S.parameters
    for n = 1:length(Sp)
        push!(t.args, Expr(:call, Expr(:curly, :StaticInt, Sp[n])))
    end
    return t
end
@generated function ArrayInterfaceCore.strides(A::StaticArrays.StaticArray{S}) where {S}
    t = Expr(:tuple, Expr(:call, Expr(:curly, :StaticInt, 1)))
    Sp = S.parameters
    x = 1
    for n = 1:(length(Sp)-1)
        push!(t.args, Expr(:call, Expr(:curly, :StaticInt, (x *= Sp[n]))))
    end
    return t
end
if StaticArrays.SizedArray{Tuple{8,8},Float64,2,2} isa UnionAll
    @inline ArrayInterfaceCore.strides(B::StaticArrays.SizedArray{S,T,M,N,A}) where {S,T,M,N,A<:SubArray} = ArrayInterfaceCore.strides(B.data)
    ArrayInterfaceCore.parent_type(::Type{<:StaticArrays.SizedArray{S,T,M,N,A}}) where {S,T,M,N,A} = A
else
    ArrayInterfaceCore.parent_type(::Type{<:StaticArrays.SizedArray{S,T,M,N}}) where {S,T,M,N} = Array{T,N}
end

Adapt.adapt_storage(::Type{<:StaticArrays.SArray{S}}, xs::Array) where {S} = SArray{S}(xs)

end # module
