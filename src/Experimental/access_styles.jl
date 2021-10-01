
"""
    AccessStyle(I)

`AccessStyle` specifies how the default index `I` accesses other collections.
"""
abstract type AccessStyle end

struct AccessElement{N} <: AccessStyle end

struct AccessUnkown{T} <: AccessStyle end

struct AccessBoolean <: AccessStyle end

struct AccessRange <: AccessStyle end

struct AccessIndices{N} <: AccessStyle end

# FIXME This should be lispy so we can have .. specialization
# _astyle(::Type{I}, i::StaticInt) where {I} = AccessStyle(_get_tuple(I, i))
# AccessStyle(::Type{I}) where {I<:Tuple) = AccessIndices(eachop(_astyle, nstatic(Val(N)), I))

@generated function static_typed_tail(::Type{T}) where {T<:Tuple}
    N = length(T.parameters)
    out = Expr(:curly, :Tuple)
    for i in 2:N
        push!(out.args, T.parameters[i])
    end
    return out
end

AccessStyle(::Type{T}) where {T} = AccessUnkown{T}()
AccessStyle(@nospecialize(x::Type{<:Integer})) = AccessElement{1}()
AccessStyle(::Type{<:Union{OneTo,UnitRange,StepRange,OptionallyStaticRange}}) = AccessRange()
@inline AccessStyle(::Type{<:SubIndex{<:Any,I}}) where {I} = AccessElement{sum(dynamic(ndims_index(I)))}()
AccessStyle(x::Type{<:StrideIndex{N,I1,I2}}) where {N,I1,I2} = AccessElement{length(I2)}()
AccessStyle(x::Type{PermutedIndex{2,(2,1),(2,1)}}) = AccessElement{1}()
AccessStyle(x::Type{<:AbstractCartesianIndex{N}}) where {N} = AccessElement{N}()
# TODO should dig into parents
AccessStyle(x::Type{<:AbstractArray}) = AccessStyle(eltype(x))

AccessStyle(::Type{Tuple{I}}) where {I} = (AccessStyle(I),)
@inline function AccessStyle(x::Type{Tuple{I,Vararg{Any}}}) where {I}
    (AccessStyle(I), AccessStyle(static_typed_tail(x))...)
end
AccessStyle(@nospecialize(x)) = AccessStyle(typeof(x))

