struct Contiguous{N} end
Base.@pure Contiguous(N::Int) = Contiguous{N}()
_get(::Contiguous{N}) where {N} = N
"""
contiguous_axis(::Type{T}) -> Contiguous{N}

Returns the axis of an array of type `T` containing contiguous data.
If no axis is contiguous, it returns `Contiguous{-1}`.
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
contiguous_axis(::Type{<:Array}) = Contiguous{1}()
contiguous_axis(::Type{<:Tuple}) = Contiguous{1}()
function contiguous_axis(::Type{<:Union{Transpose{T,A},Adjoint{T,A}}}) where {T,A<:AbstractVector{T}}
    c = contiguous_axis(A)
    isnothing(c) && return nothing
    c === Contiguous{1}() ? Contiguous{2}() : Contiguous{-1}()
end
function contiguous_axis(::Type{<:Union{Transpose{T,A},Adjoint{T,A}}}) where {T,A<:AbstractMatrix{T}}
    c = contiguous_axis(A)
    isnothing(c) && return nothing
    contig = _get(c)
    new_contig = contig == -1 ? -1 : 3 - contig
    Contiguous{new_contig}()
end
function contiguous_axis(::Type{<:PermutedDimsArray{T,N,I1,I2,A}}) where {T,N,I1,I2,A<:AbstractArray{T,N}}
    c = contiguous_axis(A)
    isnothing(c) && return nothing
    contig = _get(c)
    new_contig = contig == -1 ? -1 : I2[_get(c)]
    Contiguous{new_contig}()
end
function contiguous_axis(::Type{S}) where {N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}}
    _contiguous_axis(S, contiguous_axis(A))
end
_contiguous_axis(::Any, ::Nothing) = nothing
@generated function _contiguous_axis(::Type{S}, ::Contiguous{C}) where {C,N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}}
    n = 0
    new_contig = contig = C
    for np in 1:NP
        p = I.parameters[np]
        if p <: OrdinalRange
            n += 1
            if np == contig
                new_contig = (p <: AbstractUnitRange) ? n : -1
            end
        elseif p <: AbstractArray
            n += 1
            new_contig = np == contig ? -1 : new_contig
        elseif p <: Integer
            if np == contig
                new_contig = -1
            end
        end
    end
    # If n != N, then an axis was indexed by something other than an integer or `OrdinalRange`, so we return `nothing`.
    n == N || return nothing
    Expr(:call, Expr(:curly, :Contiguous, new_contig))
end

# contiguous_if_one(::Contiguous{1}) = Contiguous{1}()
# contiguous_if_one(::Any) = Contiguous{-1}()
function contiguous_axis(::Type{R}) where {T, N, S, A <: Array{S}, R <: Base.ReinterpretArray{T, N, S, A}}
    isbitstype(S) ? Contiguous{1}() : nothing
    # contiguous_if_one(contiguous_axis(parent_type(R)))
end


"""
    contiguous_axis_indicator(::Type{T}) -> Tuple{Vararg{<:Val}}

Returns a tuple boolean `Val`s indicating whether that axis is contiguous.
"""
contiguous_axis_indicator(::Type{A}) where {D, A <: AbstractArray{<:Any,D}} = contiguous_axis_indicator(contiguous_axis(A), Val(D))
contiguous_axis_indicator(::A) where {A <: AbstractArray} = contiguous_axis_indicator(A)
contiguous_axis_indicator(::Nothing, ::Val) = nothing
Base.@pure contiguous_axis_indicator(::Contiguous{N}, ::Val{D}) where {N,D} = ntuple(d -> Val{d == N}(), Val{D}())

"""
If the contiguous dimension is not the dimension with `StrideRank{1}`:
"""
struct ContiguousBatch{N} end
Base.@pure ContiguousBatch(N::Int) = ContiguousBatch{N}()
_get(::ContiguousBatch{N}) where {N} = N

"""
    contiguous_batch_size(::Type{T}) -> ContiguousBatch{N}

Returns the Base.size of contiguous batches if `!isone(stride_rank(T, contiguous_axis(T)))`.
If `isone(stride_rank(T, contiguous_axis(T)))`, then it will return `ContiguousBatch{0}()`.
If `contiguous_axis(T) == -1`, it will return `ContiguousBatch{-1}()`.
If unknown, it will return `nothing`.
"""
contiguous_batch_size(x) = contiguous_batch_size(typeof(x))
contiguous_batch_size(::Type) = nothing
contiguous_batch_size(::Type{Array{T,N}}) where {T,N} = ContiguousBatch{0}()
contiguous_batch_size(::Type{<:Tuple}) = ContiguousBatch{0}()
contiguous_batch_size(::Type{<:Union{Transpose{T,A},Adjoint{T,A}}}) where {T,A<:AbstractVecOrMat{T}} = contiguous_batch_size(A)
contiguous_batch_size(::Type{<:PermutedDimsArray{T,N,I1,I2,A}}) where {T,N,I1,I2,A<:AbstractArray{T,N}} = contiguous_batch_size(A)
function contiguous_batch_size(::Type{S}) where {N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}}
    _contiguous_batch_size(S, contiguous_batch_size(A), contiguous_axis(A))
end
_contiguous_batch_size(::Any, ::Any, ::Any) = nothing
@generated function _contiguous_batch_size(::Type{S}, ::ContiguousBatch{B}, ::Contiguous{C}) where {B,C,N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}}
    if I.parameters[C] <: AbstractUnitRange
        Expr(:call, Expr(:curly, :ContiguousBatch, B))
    else
        Expr(:call, Expr(:curly, :ContiguousBatch, -1))
    end
end

contiguous_batch_size(::Type{R}) where {T, N, S, A <: Array{S}, R <: Base.ReinterpretArray{T, N, S, A}} = ContiguousBatch{0}()

struct StrideRank{R} end
Base.@pure StrideRank(R::NTuple{<:Any,Int}) = StrideRank{R}()
_get(::StrideRank{R}) where {R} = R
Base.collect(::StrideRank{R}) where {R} = collect(R)
@inline Base.getindex(::StrideRank{R}, i::Integer) where {R} = R[i]
@inline Base.getindex(::StrideRank{R}, ::Val{I}) where {R,I} = StrideRank{permute(R, I)}()

"""
    rank_to_sortperm(::StrideRank) -> NTuple{N,Int}

Returns the `sortperm` of the stride ranks.
"""
function rank_to_sortperm(R::NTuple{N,Int}) where {N}
    sp = ntuple(zero, Val{N}())
    r = ntuple(n -> sum(R[n] .≥ R), Val{N}())
    @inbounds for n in 1:N
        sp = Base.setindex(sp, n, r[n])
    end
    sp
end
@generated Base.sortperm(::StrideRank{R}) where {R} = rank_to_sortperm(R)

stride_rank(x) = stride_rank(typeof(x))
stride_rank(::Type) = nothing
stride_rank(::Type{Array{T,N}}) where {T,N} = StrideRank{ntuple(identity, Val{N}())}()
stride_rank(::Type{<:Tuple}) = StrideRank{(1,)}()

stride_rank(::Type{B}) where {T, A <: AbstractVector{T}, B <: Union{Transpose{T,A},Adjoint{T,A}}} = StrideRank{(2, 1)}()
stride_rank(::Type{B}) where {T, A <: AbstractMatrix{T}, B <: Union{Transpose{T,A},Adjoint{T,A}}} = _stride_rank(B, stride_rank(A))
_stride_rank(::Type{<:Union{Transpose{T,A},Adjoint{T,A}}}, ::Nothing) where {T,A<:AbstractMatrix{T}} = nothing
_stride_rank(::Type{<:Union{Transpose{T,A},Adjoint{T,A}}}, rank) where {T,A<:AbstractMatrix{T}} = rank[Val{(2,1)}()]

stride_rank(::Type{B}) where {T,N,I1,I2,A<:AbstractArray{T,N},B<:PermutedDimsArray{T,N,I1,I2,A}} = _stride_rank(B, stride_rank(A))
_stride_rank(::Type{B}, ::Nothing) where {T,N,I1,I2,A<:AbstractArray{T,N},B<:PermutedDimsArray{T,N,I1,I2,A}} = nothing
_stride_rank(::Type{B}, rank) where {T,N,I1,I2,A<:AbstractArray{T,N},B<:PermutedDimsArray{T,N,I1,I2,A}} = rank[Val{I1}()]
function stride_rank(::Type{S}) where {N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}}
    _stride_rank(S, stride_rank(A))
end
_stride_rank(::Any, ::Any) = nothing
@generated function _stride_rank(::Type{S}, ::StrideRank{R}) where {R,N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}}
    rankv = collect(R)
    rank_new = Int[]
    n = 0
    for np in 1:NP
        r = rankv[np]
        if I.parameters[np] <: AbstractArray
            n += 1
            push!(rank_new, r)
        end
    end
    # If n != N, then an axis was indexed by something other than an integer or `AbstractUnitRange`, so we return `nothing`.
    n == N || return nothing
    ranktup = Expr(:tuple); append!(ranktup.args, rank_new) # dynamic splats bad
    Expr(:call, Expr(:curly, :StrideRank, ranktup))
end
stride_rank(x, i) = stride_rank(x)[i]
stride_rank(::Type{R}) where {T, N, S, A <: Array{S}, R <: Base.ReinterpretArray{T, N, S, A}} = StrideRank{ntuple(identity, Val{N}())}()

"""
    is_column_major(A) -> Val{true/false}()

Returns `Val{true}` if elements of `A` are stored in column major order. Otherwise returns `Val{false}`.
"""
is_column_major(A) = is_column_major(stride_rank(A), contiguous_batch_size(A))
is_column_major(::Nothing, ::Any) = Val{false}()
@generated function is_column_major(::StrideRank{R}, ::ContiguousBatch{N}) where {R,N}
    N > 0 && return :(Val{false}())
    N = length(R)
    for n ∈ 2:N
        if R[n] ≤ R[n-1]
            return :(Val{false}())
        end
    end
    :(Val{true}())
end

struct DenseDims{D} end
Base.@pure DenseDims(D::NTuple{<:Any,Bool}) = DenseDims{D}()
@inline Base.getindex(::DenseDims{D}, i::Integer) where {D} = D[i]
@inline Base.getindex(::DenseDims{D}, ::Val{I}) where {D,I} = DenseDims{permute(D, I)}()
"""
    dense_dims(::Type{T}) -> NTuple{N,Bool}

Returns a tuple of indicators for whether each axis is dense.
An axis `i` of array `A` is dense if `stride(A, i) * Base.size(A, i) == stride(A, j)` where `stride_rank(A)[i] + 1 == stride_rank(A)[j]`.
"""
dense_dims(x) = dense_dims(typeof(x))
dense_dims(::Type) = nothing
dense_dims(::Type{Array{T,N}}) where {T,N} = DenseDims{ntuple(_ -> true, Val{N}())}()
dense_dims(::Type{<:Tuple}) = DenseDims{(true,)}()
function dense_dims(::Type{<:Union{Transpose{T,A},Adjoint{T,A}}}) where {T,A<:AbstractMatrix{T}}
    dense = dense_dims(A)
    isnothing(dense) ? nothing : dense[Val{(2,1)}()]
end
function dense_dims(::Type{<:PermutedDimsArray{T,N,I1,I2,A}}) where {T,N,I1,I2,A<:AbstractArray{T,N}}
    dense = dense_dims(A)
    isnothing(dense) ? nothing : dense[Val{I1}()]
end
function dense_dims(::Type{S}) where {N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}}
    _dense_dims(S, dense_dims(A), stride_rank(A))
end
_dense_dims(::Any, ::Any) = nothing
@generated function _dense_dims(::Type{S}, ::DenseDims{D}, ::StrideRank{R}) where {D,R,N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}}
    still_dense = true
    sp = rank_to_sortperm(R)
    densev = Vector{Bool}(undef, NP)
    for np in 1:NP
        spₙ = sp[np]
        still_dense &= D[spₙ]
        densev[spₙ] = still_dense
        # a dim not being complete makes later dims not dense
        still_dense &= (I.parameters[spₙ] <: Base.Slice)::Bool
    end
    dense_tup = Expr(:tuple)
    for np in 1:NP
        Iₙₚ = I.parameters[np]
        if Iₙₚ <: AbstractUnitRange
            push!(dense_tup.args, densev[np])
        elseif Iₙₚ <: AbstractVector
            push!(dense_tup.args, false)
        end
    end
    # If n != N, then an axis was indexed by something other than an integer or `AbstractUnitRange`, so we return `nothing`.
    length(dense_tup.args) == N ? Expr(:call, Expr(:curly, :DenseDims, dense_tup)) : nothing
end

permute(t::NTuple{N}, I::NTuple{N,Int}) where {N} = ntuple(n -> t[I[n]], Val{N}())
@generated function permute(t::Tuple{Vararg{Any,N}}, ::Val{I}) where {N,I}
    t = Expr(:tuple)
    foreach(i -> push!(t.args, Expr(:ref, :t, i)), I)
    Expr(:block, Expr(:meta, :inline), t)
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

@generated function _perm_tuple(::Type{T}, ::Val{P}) where {T,P}
    out = Expr(:curly, :Tuple)
    for p in P
        push!(out.args, :(T.parameters[$p]))
    end
    Expr(:block, Expr(:meta, :inline), out)
end

"""
    axes_types(::Type{T}[, d]) -> Type

Returns the type of the axes for `T`
"""
axes_types(x) = axes_types(typeof(x))
axes_types(x, d) = axes_types(typeof(x), d)
@inline axes_types(::Type{T}, d) where {T} = axes_types(T).parameters[to_dims(T, d)]
function axes_types(::Type{T}) where {T}
    if parent_type(T) <: T
        return Tuple{Vararg{OptionallyStaticUnitRange{One,Int},ndims(T)}}
    else
        return axes_types(parent_type(T))
    end
end
axes_types(::Type{T}) where {T<:Adjoint} = _perm_tuple(axes_types(parent_type(T)), Val((2, 1)))
axes_types(::Type{T}) where {T<:Transpose} = _perm_tuple(axes_types(parent_type(T)), Val((2, 1)))
function axes_types(::Type{T}) where {I1,T<:PermutedDimsArray{<:Any,<:Any,I1}}
    return _perm_tuple(axes_types(parent_type(T)), Val(I1))
end
function axes_types(::Type{T}) where {T<:OptionallyStaticRange}
    if known_length(T) === nothing
        return Tuple{OptionallyStaticUnitRange{One,Int}}
    else
        return Tuple{OptionallyStaticUnitRange{One,StaticInt{known_length(T) - 1}}}
    end
end

@inline function axes_types(::Type{T}) where {P,I,T<:SubArray{<:Any,<:Any,P,I}}
    return _sub_axes_types(Val(ArrayStyle(T)), I, axes_types(P))
end
@generated function _sub_axes_types(::Val{S}, ::Type{I}, ::Type{PI}) where {S,I<:Tuple,PI<:Tuple}
    out = Expr(:curly, :Tuple)
    d = 1
    for i in I.parameters
        ad = argdims(S, i)
        if ad > 0
            push!(out.args, :(sub_axis_type($(PI.parameters[d]), $i)))
            d += ad
        else
            d += 1
        end
    end
    Expr(:block, Expr(:meta, :inline), out)
end

@inline function axes_types(::Type{T}) where {T<:Base.ReinterpretArray}
    return _reinterpret_axes_types(axes_type(parent_type(T)), eltype(T), eltype(parent_type(T)))
end
@generated function _reinterpret_axes_types(::Type{I}, ::Type{T}, ::Type{S}) where {I<:Tuple,T,S}
    out = Expr(:curly, :Tuple)
    for i in 1:length(T.parameters)
        if i === 1
            push!(out.args, :(reinterpret_axis_type($(I.parameters[1]), $T, $S)))
        else
            # FIXME double check this once I've slept
            push!(out.args, :($(I.parameters[i])))
        end
    end
    Expr(:block, Expr(:meta, :inline), out)
end


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
        return OptionallyStaticUnitRange{One,StaticInt{Int(known_length(A) / (sizeof(T) / sizeof(S))) - 1}}
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
    return permute(known_strides(parent_type(T)), Val{(2,1)}())
end
function known_strides(::Type{T}) where {T<:Transpose}
    return permute(known_strides(parent_type(T)), Val{(2,1)}())
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

#=

function strides(a::ReinterpretArray)
    a.parent isa StridedArray || ArgumentError("Parent must be strided.") |> throw
    size_to_strides(1, size(a)...)
end

strides(a::Union{DenseArray,StridedReshapedArray,StridedReinterpretArray}) = size_to_strides(1, size(a)...)
    @generated function _strides(_::Base.ReinterpretArray{T, N, S, A, true}, s::NTuple{N}, ::Contiguous{1}) where {T, N, S, D, A <: Array{S,D}}
        stup = Expr(:tuple, :(One()))
        if D < N
            push!(stup.args, Expr(:call, Expr(:curly, :StaticInt, sizeof(S) ÷ sizeof(T))))
        end
        for n ∈ 2+(D < N):N
            push!(stup.args, Expr(:ref, :s, n))
        end
        quote
            $(Expr(:meta,:inline))
            @inbounds $stup
        end
    end
    =#

"""
    offsets(A) -> Tuple

Returns offsets of indices with respect to 0. If values are known at compile time,
it should return them as `Static` numbers.
For example, if `A isa Base.Matrix`, `offsets(A) === (StaticInt(1), StaticInt(1))`.
"""
offsets(::Any) = (StaticInt{1}(),) # Assume arbitrary Julia data structures use 1-based indexing by default.
@inline strides(A::Vector{<:Any}) = (StaticInt(1),)
@inline strides(A::Array{<:Any,N}) where {N} = (StaticInt(1), Base.tail(Base.strides(A))...)
@inline strides(A::AbstractArray) = _strides(A, Base.strides(A), contiguous_axis(A))

@inline function strides(x::LinearAlgebra.Adjoint{T,V}) where {T, V <: AbstractVector{T}}
    strd = stride(parent(x), One())
    (strd, strd)
end
@inline function strides(x::LinearAlgebra.Transpose{T,V}) where {T, V <: AbstractVector{T}}
    strd = stride(parent(x), One())
    (strd, strd)
end

@generated function _strides(A::AbstractArray{T,N}, s::NTuple{N}, ::Contiguous{C}) where {T,N,C}
    if C ≤ 0 || C > N
        return Expr(:block, Expr(:meta,:inline), :s)
    end
    stup = Expr(:tuple)
    for n ∈ 1:N
        if n == C
            push!(stup.args, :(One()))
        else
            push!(stup.args, Expr(:ref, :s, n))
        end
    end
    quote
        $(Expr(:meta,:inline))
        @inbounds $stup
    end
end

if VERSION ≥ v"1.6.0-DEV.1581"
    @generated function _strides(_::Base.ReinterpretArray{T, N, S, A, true}, s::NTuple{N}, ::Contiguous{1}) where {T, N, S, D, A <: Array{S,D}}
        stup = Expr(:tuple, :(One()))
        if D < N
            push!(stup.args, Expr(:call, Expr(:curly, :StaticInt, sizeof(S) ÷ sizeof(T))))
        end
        for n ∈ 2+(D < N):N
            push!(stup.args, Expr(:ref, :s, n))
        end
        quote
            $(Expr(:meta,:inline))
            @inbounds $stup
        end
    end
end

@inline offsets(x, i) = static_first(indices(x, i))
# @inline offsets(A::AbstractArray{<:Any,N}) where {N} = ntuple(n -> offsets(A, n), Val{N}())
# Explicit tuple needed for inference.
@generated function offsets(A::AbstractArray{<:Any,N}) where {N}
    t = Expr(:tuple)
    for n ∈ 1:N
        push!(t.args, :(offsets(A, StaticInt{$n}())))
    end
    Expr(:block, Expr(:meta, :inline), t)
end

@inline size(B::Union{Transpose{T,A},Adjoint{T,A}}) where {T,A<:AbstractMatrix{T}} = permute(size(parent(B)), Val{(2,1)}())
@inline size(B::PermutedDimsArray{T,N,I1,I2,A}) where {T,N,I1,I2,A<:AbstractArray{T,N}} = permute(size(parent(B)), Val{I1}())
@inline size(A::AbstractArray, ::StaticInt{N}) where {N} = size(A)[N]
@inline size(A::AbstractArray, ::Val{N}) where {N} = size(A)[N]
@inline strides(B::Union{Transpose{T,A},Adjoint{T,A}}) where {T,A<:AbstractMatrix{T}} = permute(strides(parent(B)), Val{(2,1)}())
@inline strides(B::PermutedDimsArray{T,N,I1,I2,A}) where {T,N,I1,I2,A<:AbstractArray{T,N}} = permute(strides(parent(B)), Val{I1}())
@inline stride(A::AbstractArray, ::StaticInt{N}) where {N} = strides(A)[N]
@inline stride(A::AbstractArray, ::Val{N}) where {N} = strides(A)[N]
stride(A, i) = Base.stride(A, i) # for type stability

size(B::S) where {N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}} = _size(size(parent(B)), B.indices, map(static_length, B.indices))
strides(B::S) where {N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}} = _strides(strides(parent(B)), B.indices)
@generated function _size(A::Tuple{Vararg{Any,N}}, inds::I, l::L) where {N, I<:Tuple, L}
    t = Expr(:tuple)
    for n in 1:N
        if (I.parameters[n] <: Base.Slice)
            push!(t.args, :(@inbounds(_try_static(A[$n], l[$n]))))
        elseif I.parameters[n] <: Number
            nothing
        else
            push!(t.args, Expr(:ref, :l, n))
        end
    end
    Expr(:block, Expr(:meta, :inline), t)
end
@generated function _strides(A::Tuple{Vararg{Any,N}}, inds::I) where {N, I<:Tuple}
    t = Expr(:tuple)
    for n in 1:N
        if I.parameters[n] <: AbstractUnitRange
            push!(t.args, Expr(:ref, :A, n))
        elseif I.parameters[n] <: AbstractRange
            push!(t.args, Expr(:call, :(*), Expr(:ref, :A, n), Expr(:call, :static_step, Expr(:ref, :inds, n))))
        elseif !(I.parameters[n] <: Integer)
            return nothing
        end
    end
    Expr(:block, Expr(:meta, :inline), t)
end

