
"""
    offsets(A) -> Tuple

Returns offsets of indices with respect to 0. If values are known at compile time,
it should return them as `Static` numbers.
For example, if `A isa Base.Matrix`, `offsets(A) === (StaticInt(1), StaticInt(1))`.
"""
@inline offsets(x, i) = static_first(indices(x, i))
# Explicit tuple needed for inference.
offsets(x) = each_op_x(offsets, x)
offsets(::Tuple) = (One(),)

"""
contiguous_axis(::Type{T}) -> StaticInt{N}

Returns the axis of an array of type `T` containing contiguous data.
If no axis is contiguous, it returns `StaticInt{-1}`.
If unknown, it returns `nothing`.
"""
contiguous_axis(x) = contiguous_axis(typeof(x))
function contiguous_axis(::Type{T}) where {T}
    if parent_type(T) <: T
        return nothing
    else
        return contiguous_axis(parent_type(T))
    end
end
contiguous_axis(::Type{<:Array}) = StaticInt{1}()
contiguous_axis(::Type{<:Tuple}) = StaticInt{1}()
function contiguous_axis(::Type{T}) where {T<:VecAdjTrans}
    c = contiguous_axis(parent_type(T))
    if c === nothing
        return nothing
    elseif c === One()
        return StaticInt{2}()
    else
        return -c
    end
end
function contiguous_axis(::Type{T}) where {T<:MatAdjTrans}
    c = contiguous_axis(parent_type(T))
    if c === nothing
        return nothing
    elseif isone(-c)
        return c
    else
        return StaticInt(3) - c
    end
end
function contiguous_axis(::Type{T}) where {I1,I2,T<:PermutedDimsArray{<:Any,<:Any,I1,I2}}
    c = contiguous_axis(parent_type(T))
    if c === nothing
        return nothing
    elseif isone(-c)
        return c
    else
        return StaticInt(I2[c])
    end
end
function contiguous_axis(::Type{S}) where {N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}}
    return _contiguous_axis(S, contiguous_axis(A))
end

_contiguous_axis(::Any, ::Nothing) = nothing
function _contiguous_axis(::Type{A}, c::StaticInt{C}) where {T,N,P,I,A<:SubArray{T,N,P,I},C}
    if I.parameters[C] <: AbstractUnitRange
        return from_parent_dims(A)[C]
    elseif I.parameters[C] <: AbstractArray
        return -One()
    elseif I.parameters[C] <: Integer
        return -One()
    else
        return nothing
    end
end

# contiguous_if_one(::StaticInt{1}) = StaticInt{1}()
# contiguous_if_one(::Any) = StaticInt{-1}()
function contiguous_axis(::Type{R}) where {T,N,S,A<:Array{S},R<:ReinterpretArray{T,N,S,A}}
    if isbitstype(S)
        return One()
    else
        return nothing
    end
end

"""
    contiguous_axis_indicator(::Type{T}) -> Tuple{Vararg{Val}}

Returns a tuple boolean `Val`s indicating whether that axis is contiguous.
"""
function contiguous_axis_indicator(::Type{A}) where {D,A<:AbstractArray{<:Any,D}}
    return contiguous_axis_indicator(contiguous_axis(A), Val(D))
end
contiguous_axis_indicator(::A) where {A<:AbstractArray} = contiguous_axis_indicator(A)
contiguous_axis_indicator(::Nothing, ::Val) = nothing
Base.@pure function contiguous_axis_indicator(::StaticInt{N}, ::Val{D}) where {N,D}
    return ntuple(d -> StaticBool(d === N), Val{D}())
end

function rank_to_sortperm(R::Tuple{Vararg{StaticInt,N}}) where {N}
    sp = ntuple(zero, Val{N}())
    r = ntuple(n -> sum(R[n] .≥ R), Val{N}())
    @inbounds for n = 1:N
        sp = Base.setindex(sp, n, r[n])
    end
    sp
end

stride_rank(x) = stride_rank(typeof(x))
stride_rank(::Type) = nothing
stride_rank(::Type{Array{T,N}}) where {T,N} = nstatic(Val(N))
stride_rank(::Type{<:Tuple}) = (One(),)

stride_rank(::Type{T}) where {T<:VecAdjTrans} = (StaticInt(2), StaticInt(1))
stride_rank(::Type{T}) where {T<:MatAdjTrans} = _stride_rank(T, stride_rank(parent_type(T)))
_stride_rank(::Type{T}, ::Nothing) where {T<:MatAdjTrans} = nothing
_stride_rank(::Type{T}, rank) where {T<:MatAdjTrans} = (last(rank), first(rank))

function stride_rank(::Type{T},) where {T<:PermutedDimsArray}
    return _stride_rank(T, stride_rank(parent_type(T)))
end
_stride_rank(::Type{T}, ::Nothing) where {T<:PermutedDimsArray} = nothing
function _stride_rank(::Type{T}, rank) where {I,T<:PermutedDimsArray{<:Any,<:Any,I}}
    return permute(rank, Val(I))
end

function stride_rank(::Type{S}) where {N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}}
    return _stride_rank(S, stride_rank(A))
end
_stride_rank(::Any, ::Any) = nothing
@generated function _stride_rank(
    ::Type{S},
    ::R,
) where {R,N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}}
    rank_new = []
    n = 0
    for np = 1:NP
        r = R.parameters[np].parameters[1]
        if I.parameters[np] <: AbstractArray
            n += 1
            push!(rank_new, :(StaticInt($r)))
        end
    end
    # If n != N, then an axis was indexed by something other than an integer or `AbstractUnitRange`, so we return `nothing`.
    n == N || return nothing
    ranktup = Expr(:tuple)
    append!(ranktup.args, rank_new) # dynamic splats bad
    return ranktup
end
stride_rank(x, i) = stride_rank(x)[i]
function stride_rank(::Type{R}) where {T,N,S,A<:Array{S},R<:Base.ReinterpretArray{T,N,S,A}}
    return nstatic(Val(N))
end

function stride_rank(::Type{Base.ReshapedArray{T, N, P, Tuple{Vararg{Base.SignedMultiplicativeInverse{Int},M}}}}) where {T,N,P,M}
    _reshaped_striderank(is_column_major(P), Val{N}(), Val{M}())
end
_reshaped_striderank(::True, ::Val{N}, ::Val{0}) where {N} = nstatic(Val(N))
_reshaped_striderank(_, __, ___) = nothing


"""
If the contiguous dimension is not the dimension with `StrideRank{1}`:
"""

"""
    contiguous_batch_size(::Type{T}) -> StaticInt{N}

Returns the Base.size of contiguous batches if `!isone(stride_rank(T, contiguous_axis(T)))`.
If `isone(stride_rank(T, contiguous_axis(T)))`, then it will return `StaticInt{0}()`.
If `contiguous_axis(T) == -1`, it will return `StaticInt{-1}()`.
If unknown, it will return `nothing`.
"""
contiguous_batch_size(x) = contiguous_batch_size(typeof(x))
contiguous_batch_size(::Type{T}) where {T} = _contiguous_batch_size(contiguous_axis(T), stride_rank(T))
_contiguous_batch_size(_, __) = nothing
function _contiguous_batch_size(::StaticInt{D}, ::R) where {D,R<:Tuple}
    if R.parameters[D].parameters[1] === 1
        return Zero()
    else
        return nothing
    end
end

contiguous_batch_size(::Type{Array{T,N}}) where {T,N} = StaticInt{0}()
contiguous_batch_size(::Type{<:Tuple}) = StaticInt{0}()
function contiguous_batch_size(::Type{T}) where {T<:Union{Transpose,Adjoint}}
    return contiguous_batch_size(parent_type(T))
end
function contiguous_batch_size(::Type{T}) where {T<:PermutedDimsArray}
    return contiguous_batch_size(parent_type(T))
end
function contiguous_batch_size(::Type{S}) where {N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}}
    return _contiguous_batch_size(S, contiguous_batch_size(A), contiguous_axis(A))
end
_contiguous_batch_size(::Any, ::Any, ::Any) = nothing
function _contiguous_batch_size(::Type{<:SubArray{T,N,A,I}}, b::StaticInt{B}, c::StaticInt{C}) where {T,N,A,I,B,C}
    if I.parameters[C] <: AbstractUnitRange
        return b
    else
        return -One()
    end
end
contiguous_batch_size(::Type{<:Base.ReinterpretArray{T,N,S,A}}) where {T,N,S,A} = Zero()

"""
    is_column_major(A) -> True/False

Returns `Val{true}` if elements of `A` are stored in column major order. Otherwise returns `Val{false}`.
"""
is_column_major(A) = is_column_major(stride_rank(A), contiguous_batch_size(A))
is_column_major(sr::Nothing, cbs) = False()
is_column_major(sr::R, cbs) where {R} = _is_column_major(sr, cbs)

# cbs > 0
_is_column_major(sr::R, cbs::StaticInt) where {R} = False()
# cbs <= 0
_is_column_major(sr::R, cbs::Union{StaticInt{0},StaticInt{-1}}) where {R} = is_increasing(sr)

"""
    dense_dims(::Type{T}) -> NTuple{N,Bool}

Returns a tuple of indicators for whether each axis is dense.
An axis `i` of array `A` is dense if `stride(A, i) * Base.size(A, i) == stride(A, j)` where `stride_rank(A)[i] + 1 == stride_rank(A)[j]`.
"""
dense_dims(x) = dense_dims(typeof(x))
dense_dims(::Type) = nothing
_all_dense(::Val{N}) where {N} = ntuple(_ -> True(), Val{N}())

dense_dims(::Type{Array{T,N}}) where {T,N} = _all_dense(Val{N}())
dense_dims(::Type{<:Tuple}) = (True(),)
function dense_dims(::Type{T}) where {T<:MatAdjTrans}
    dense = dense_dims(parent_type(T))
    if dense === nothing
        return nothing
    else
        return (last(dense), first(dense))
    end
end
function dense_dims(::Type{<:PermutedDimsArray{T,N,I1,I2,A}}) where {T,N,I1,I2,A}
    dense = dense_dims(A)
    if dense === nothing
        return nothing
    else
        return permute(dense, Val(I1))
    end
end
function dense_dims(::Type{S}) where {N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}}
    return _dense_dims(S, dense_dims(A), Val(stride_rank(A))) # TODO fix this
end

_dense_dims(::Any, ::Any) = nothing
@generated function _dense_dims(
    ::Type{S},
    ::D,
    ::Val{R},
) where {D,R,N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}}
    still_dense = true
    sp = rank_to_sortperm(R)
    densev = Vector{Bool}(undef, NP)
    for np = 1:NP
        spₙ = sp[np]
        if still_dense
            still_dense = D.parameters[spₙ] <: True
        end
        densev[spₙ] = still_dense
        # a dim not being complete makes later dims not dense
        still_dense &= (I.parameters[spₙ] <: Base.Slice)::Bool
    end
    dense_tup = Expr(:tuple)
    for np = 1:NP
        Iₙₚ = I.parameters[np]
        if Iₙₚ <: AbstractUnitRange
            if densev[np]
                push!(dense_tup.args, :(True()))
            else
                push!(dense_tup.args, :(False()))
            end
        elseif Iₙₚ <: AbstractVector
            push!(dense_tup.args, :(False()))
        end
    end
    # If n != N, then an axis was indexed by something other than an integer or `AbstractUnitRange`, so we return `nothing`.
    if length(dense_tup.args) === N
        return dense_tup
    else
        return nothing
    end
end

function dense_dims(::Type{Base.ReshapedArray{T, N, P, Tuple{Vararg{Base.SignedMultiplicativeInverse{Int},M}}}}) where {T,N,P,M}
    return _reshaped_dense_dims(dense_dims(P), is_column_major(P), Val{N}(), Val{M}())
end
_reshaped_dense_dims(_, __, ___, ____) = nothing
function _reshaped_dense_dims(dense::D, ::True, ::Val{N}, ::Val{0}) where {D,N}
    if all(dense)
        return _all_dense(Val{N}())
    else
        return nothing
    end
end

"""
    strides(A) -> Tuple

Returns the strides of array `A`. If any strides are known at compile time,
these should be returned as `Static` numbers. For example:
```julia
julia> A = rand(3,4);

julia> ArrayInterface.strides(A)
(StaticInt{1}(), 3)

Additionally, the behavior differs from `Base.strides` for adjoint vectors:

julia> x = rand(5);

julia> ArrayInterface.strides(x')
(StaticInt{1}(), StaticInt{1}())

This is to support the pattern of using just the first stride for linear indexing, `x[i]`,
while still producing correct behavior when using valid cartesian indices, such as `x[1,i]`.
```
"""
strides(A) = Base.strides(A)
strides(A, d) = strides(A)[to_dims(A, d)]

@inline function known_length(::Type{T}) where {T <: Base.ReinterpretArray}
    return _known_length(known_length(parent_type(T)), eltype(T), eltype(parent_type(T)))
end
_known_length(::Nothing, _, __) = nothing
@inline _known_length(L::Integer, ::Type{T}, ::Type{P}) where {T,P} = L * sizeof(P) ÷ sizeof(T)

# These methods help handle identifying axes that dont' directly propagate from the
# parent array axes. They may be worth making a formal part of the API, as they provide
# a low traffic spot to change what axes_types produces.
@inline function sub_axis_type(::Type{A}, ::Type{I}) where {A,I}
    if known_length(I) === nothing
        return OptionallyStaticUnitRange{One,Int}
    else
        return OptionallyStaticUnitRange{One,StaticInt{known_length(I)}}
    end
end

@inline function reinterpret_axis_type(::Type{A}, ::Type{T}, ::Type{S}) where {A,T,S}
    if known_length(A) === nothing
        return OptionallyStaticUnitRange{One,Int}
    else
        return OptionallyStaticUnitRange{
            One,
            StaticInt{Int(known_length(A) / (sizeof(T) / sizeof(S)))},
        }
    end
end

"""
    known_offsets(::Type{T}[, d]) -> Tuple

Returns a tuple of offset values known at compile time. If the offset of a given axis is
not known at compile time `nothing` is returned its position.
"""
@inline known_offsets(x, d) = known_offsets(x)[to_dims(x, d)]
known_offsets(x) = known_offsets(typeof(x))
@generated function known_offsets(::Type{T}) where {T}
    out = Expr(:tuple)
    for p in axes_types(T).parameters
        push!(out.args, known_first(p))
    end
    return out
end

"""
    known_size(::Type{T}[, d]) -> Tuple

Returns the size of each dimension for `T` known at compile time. If a dimension does not
have a known size along a dimension then `nothing` is returned in its position.
"""
@inline known_size(x, d) = known_size(x)[to_dims(x, d)]
known_size(x) = known_size(typeof(x))
known_size(::Type{T}) where {T} = _known_size(axes_types(T))
@generated function _known_size(::Type{Axs}) where {Axs<:Tuple}
    out = Expr(:tuple)
    for p in Axs.parameters
        push!(out.args, :(known_length($p)))
    end
    return Expr(:block, Expr(:meta, :inline), out)
end

"""
    known_strides(::Type{T}[, d]) -> Tuple

Returns the strides of array `A` known at compile time. Any strides that are not known at
compile time are represented by `nothing`.
"""
known_strides(x) = known_strides(typeof(x))
known_strides(x, d) = known_strides(x)[to_dims(x, d)]
known_strides(::Type{T}) where {T<:Vector} = (1,)
@inline function known_strides(::Type{T}) where {T<:Adjoint{<:Any,<:AbstractVector}}
    strd = first(known_strides(parent_type(T)))
    return (strd, strd)
end
function known_strides(::Type{T}) where {T<:Adjoint}
    return permute(known_strides(parent_type(T)), Val{(2, 1)}())
end
function known_strides(::Type{T}) where {T<:Transpose}
    return permute(known_strides(parent_type(T)), Val{(2, 1)}())
end
@inline function known_strides(::Type{T}) where {T<:Transpose{<:Any,<:AbstractVector}}
    strd = first(known_strides(parent_type(T)))
    return (strd, strd)
end
@inline function known_strides(::Type{T}) where {I1,T<:PermutedDimsArray{<:Any,<:Any,I1}}
    return permute(known_strides(parent_type(T)), Val{I1}())
end
@inline function known_strides(::Type{T}) where {I1,T<:SubArray{<:Any,<:Any,<:Any,I1}}
    return _sub_strides(Val(ArrayStyle(T)), I1, Val(known_strides(parent_type(T))))
end

@generated function _sub_strides(::Val{S}, ::Type{I}, ::Val{P}) where {S,I<:Tuple,P}
    out = Expr(:tuple)
    d = 1
    for i in I.parameters
        ad = argdims(S, i)
        if ad > 0
            push!(out.args, P[d])
            d += ad
        else
            d += 1
        end
    end
    Expr(:block, Expr(:meta, :inline), out)
end

function known_strides(::Type{T}) where {T}
    if ndims(T) === 1
        return (1,)
    else
        return _known_strides(Val(Base.front(known_size(T))))
    end
end
@generated function _known_strides(::Val{S}) where {S}
    out = Expr(:tuple)
    N = length(S)
    push!(out.args, 1)
    for s in S
        if s === nothing || out.args[end] === nothing
            push!(out.args, nothing)
        else
            push!(out.args, out.args[end] * s)
        end
    end
    return Expr(:block, Expr(:meta, :inline), out)
end

@inline strides(A::Vector{<:Any}) = (StaticInt(1),)
@inline strides(A::Array{<:Any,N}) where {N} = (StaticInt(1), Base.tail(Base.strides(A))...)
@inline strides(A::AbstractArray) = _strides(A, Base.strides(A), contiguous_axis(A))

@inline function strides(x::LinearAlgebra.Adjoint{T,V}) where {T,V<:AbstractVector{T}}
    strd = stride(parent(x), One())
    return (strd, strd)
end
@inline function strides(x::LinearAlgebra.Transpose{T,V}) where {T,V<:AbstractVector{T}}
    strd = stride(parent(x), One())
    return (strd, strd)
end

@generated function _strides(A::AbstractArray{T,N}, s::NTuple{N}, ::StaticInt{C}) where {T,N,C}
    if C ≤ 0 || C > N
        return Expr(:block, Expr(:meta, :inline), :s)
    else
        stup = Expr(:tuple)
        for n ∈ 1:N
            if n == C
                push!(stup.args, :(One()))
            else
                push!(stup.args, Expr(:ref, :s, n))
            end
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds $stup
        end
    end
end

if VERSION ≥ v"1.6.0-DEV.1581"
    @generated function _strides(
        _::Base.ReinterpretArray{T,N,S,A,true},
        s::NTuple{N},
        ::StaticInt{1},
    ) where {T,N,S,D,A<:Array{S,D}}
        stup = Expr(:tuple, :(One()))
        if D < N
            push!(stup.args, Expr(:call, Expr(:curly, :StaticInt, sizeof(S) ÷ sizeof(T))))
        end
        for n ∈ 2+(D<N):N
            push!(stup.args, Expr(:ref, :s, n))
        end
        quote
            $(Expr(:meta, :inline))
            @inbounds $stup
        end
    end
end

@inline strides(B::MatAdjTrans) = permute(strides(parent(B)), Val{(2, 1)}())
@inline function strides(B::PermutedDimsArray{T,N,I1,I2}) where {T,N,I1,I2}
    return permute(strides(parent(B)), Val{I1}())
end
@inline stride(A::AbstractArray, ::StaticInt{N}) where {N} = strides(A)[N]
@inline stride(A::AbstractArray, ::Val{N}) where {N} = strides(A)[N]
stride(A, i) = Base.stride(A, i) # for type stability

@generated function _strides(A::Tuple{Vararg{Any,N}}, inds::I) where {N,I<:Tuple}
    t = Expr(:tuple)
    for n = 1:N
        if I.parameters[n] <: AbstractUnitRange
            push!(t.args, Expr(:ref, :A, n))
        elseif I.parameters[n] <: AbstractRange
            push!(
                t.args,
                Expr(
                    :call,
                    :(*),
                    Expr(:ref, :A, n),
                    Expr(:call, :static_step, Expr(:ref, :inds, n)),
                ),
            )
        elseif !(I.parameters[n] <: Integer)
            return nothing
        end
    end
    Expr(:block, Expr(:meta, :inline), t)
end

