
"""
    known_offsets(::Type{T}) -> Tuple
    known_offsets(::Type{T}, dim) -> Union{Int,Nothing}

Returns a tuple of offset values known at compile time. If the offset of a given axis is
not known at compile time `nothing` is returned its position.
"""
known_offsets(x, dim) = known_offsets(typeof(x), dim)
known_offsets(::Type{T}, dim) where {T} = known_offsets(T, to_dims(T, dim))
known_offsets(@nospecialize T::Type{<:Number}) = ()  # Int has no dimensions
@inline function known_offsets(@nospecialize T::Type{<:SubArray})
    flatten_tuples(map_tuple_type(known_offsets, fieldtype(T, :indices)))
end
function known_offsets(::Type{T}, dim::IntType) where {T}
    if ndims(T) < dim
        return 1
    else
        return known_offsets(T)[dim]
    end
end

known_offsets(x) = known_offsets(typeof(x))
function known_offsets(::Type{T}) where {T}
    eachop(_known_offsets, ntuple(static, StaticInt(ndims(T))), axes_types(T))
end
_known_offsets(::Type{T}, dim::StaticInt) where {T} = known_first(field_type(T, dim))
known_offsets(::Type{<:StrideIndex{N,R,C,S,O}}) where {N,R,C,S,O} = known(O)

"""
    offsets(A) -> Tuple
    offsets(A, dim) -> Union{Int,StaticInt}

Returns offsets of indices with respect to 0. If values are known at compile time,
it should return them as `Static` numbers.
For example, if `A isa Base.Matrix`, `offsets(A) === (StaticInt(1), StaticInt(1))`.
"""
@inline offsets(x, i) = static_first(indices(x, i))
offsets(::Tuple) = (One(),)
offsets(x::StrideIndex) = getfield(x, :offsets)
@inline offsets(x::SubArray) = flatten_tuples(map(offsets, x.indices))
offsets(x) = eachop(_offsets, ntuple(static, StaticInt(ndims(x))), x)
function _offsets(x::X, dim::StaticInt{D}) where {X,D}
    start = known_first(axes_types(X, dim))
    if start === nothing
        return first(static_axes(x, dim))
    else
        return static(start)
    end
end
# we can't generate an axis for `StrideIndex` so this is performed manually here
@inline offsets(x::StrideIndex, dim::Int) = getfield(offsets(x), dim)
@inline offsets(x::StrideIndex, ::StaticInt{dim}) where {dim} = getfield(offsets(x), dim)

"""
    known_offset1(::Type{T}) -> Union{Int,Nothing}

Returns the linear offset of array `x` if known at compile time.
"""
@inline known_offset1(x) = known_offset1(typeof(x))
@inline function known_offset1(::Type{T}) where {T}
    if ndims(T) === 0
        return 1
    else
        return known_offsets(T, 1)
    end
end

"""
    offset1(x) -> Union{Int,StaticInt}

Returns the offset of the linear indices for `x`.
"""
@inline function offset1(x::X) where {X}
    o1 = known_offset1(X)
    if o1 === nothing
        if ndims(X) === 0
            return 1
        else
            return offsets(x, 1)
        end
    else
        return static(o1)
    end
end

"""
    contiguous_axis(::Type{T}) -> StaticInt{N}

Returns the axis of an array of type `T` containing contiguous data.
If no axis is contiguous, it returns a `StaticInt{-1}`.
If unknown, it returns `nothing`.
"""
contiguous_axis(x) = contiguous_axis(typeof(x))
contiguous_axis(::Type{<:StrideIndex{N,R,C}}) where {N,R,C} = static(C)
contiguous_axis(::Type{<:StrideIndex{N,R,nothing}}) where {N,R} = nothing
function contiguous_axis(::Type{T}) where {T}
    is_forwarding_wrapper(T) ? contiguous_axis(parent_type(T)) : nothing
end
contiguous_axis(@nospecialize T::Type{<:DenseArray}) = One()
contiguous_axis(::Type{<:BitArray}) = One()
contiguous_axis(@nospecialize T::Type{<:AbstractRange}) = One()
contiguous_axis(@nospecialize T::Type{<:Tuple}) = One()
function contiguous_axis(@nospecialize T::Type{<:VecAdjTrans})
    c = contiguous_axis(parent_type(T))
    if c === nothing
        return nothing
    elseif c === One()
        return StaticInt{2}()
    else
        return -One()
    end
end
function contiguous_axis(@nospecialize T::Type{<:MatAdjTrans})
    c = contiguous_axis(parent_type(T))
    if c === nothing
        return nothing
    elseif isone(-c)
        return c
    else
        return StaticInt(3) - c
    end
end
function contiguous_axis(@nospecialize T::Type{<:PermutedDimsArray})
    c = contiguous_axis(parent_type(T))
    if c === nothing
        return nothing
    elseif isone(-c)
        return c
    else
        return from_parent_dims(T)[c]
    end
end
function contiguous_axis(::Type{<:Base.ReshapedArray{T, N, A, Tuple{}}}) where {T, N, A}
    c = contiguous_axis(A)
    if c !== nothing && isone(-c)
        return StaticInt(-1)
    elseif dynamic(is_column_major(A) & is_dense(A))
        return StaticInt(1)
    else
        return nothing
    end
end
# we're actually looking at the original vector indices before transpose/adjoint
function contiguous_axis(::Type{<:Base.ReshapedArray{T, 1, A, Tuple{}}}) where {T, A <: VecAdjTrans}
    contiguous_axis(parent_type(A))
end
@inline function contiguous_axis(@nospecialize T::Type{<:SubArray})
    c = contiguous_axis(parent_type(T))
    if c === nothing
        return nothing
    elseif c == -1
        return c
    else
        I = field_type(fieldtype(T, :indices), c)
        if I <: AbstractUnitRange
            return from_parent_dims(T)[c]  # FIXME get rid of from_parent_dims
        elseif I <: AbstractArray || I <: IntType
            return StaticInt(-1)
        else
            return nothing
        end
    end
end

function contiguous_axis(::Type{R}) where {T,N,S,A<:Array{S},R<:ReinterpretArray{T,N,S,A}}
    if isbitstype(S)
        return One()
    else
        return nothing
    end
end

"""
    contiguous_axis_indicator(::Type{T}) -> Tuple{Vararg{StaticBool}}

Returns a tuple boolean `Val`s indicating whether that axis is contiguous.
"""
function contiguous_axis_indicator(::Type{A}) where {D,A<:AbstractArray{<:Any,D}}
    return contiguous_axis_indicator(contiguous_axis(A), Val(D))
end
contiguous_axis_indicator(::A) where {A<:AbstractArray} = contiguous_axis_indicator(A)
contiguous_axis_indicator(::Nothing, ::Val) = nothing
function contiguous_axis_indicator(c::StaticInt{N}, dim::Val{D}) where {N,D}
    map(eq(c), ntuple(static, dim))
end

function rank_to_sortperm(R::Tuple{Vararg{StaticInt,N}}) where {N}
    sp = ntuple(zero, Val{N}())
    r = ntuple(n -> sum(R[n] .≥ R), Val{N}())
    @inbounds for n = 1:N
        sp = Base.setindex(sp, n, r[n])
    end
    return sp
end

stride_rank(::Type{<:StrideIndex{N,R}}) where {N,R} = static(R)
stride_rank(x) = stride_rank(typeof(x))
function stride_rank(::Type{T}) where {T}
    is_forwarding_wrapper(T) ? stride_rank(parent_type(T)) : nothing
end
stride_rank(::Type{BitArray{N}}) where {N} = ntuple(static, StaticInt(N))
stride_rank(@nospecialize T::Type{<:DenseArray}) = ntuple(static, StaticInt(ndims(T)))
stride_rank(@nospecialize T::Type{<:AbstractRange}) = (One(),)
stride_rank(@nospecialize T::Type{<:Tuple}) = (One(),)
stride_rank(@nospecialize T::Type{<:VecAdjTrans}) = (StaticInt(2), StaticInt(1))
@inline function stride_rank(@nospecialize T::Type{<:Union{MatAdjTrans,PermutedDimsArray}})
    rank = stride_rank(parent_type(T))
    if rank === nothing
        return nothing
    else
        return map(GetIndex{false}(rank), to_parent_dims(T))
    end
 end

function stride_rank(@nospecialize T::Type{<:SubArray})
    rank = stride_rank(parent_type(T))
    if rank === nothing
        return nothing
    else
        return map(GetIndex{false}(rank), to_parent_dims(T))
    end
end

stride_rank(x, i) = stride_rank(x)[i]
function stride_rank(::Type{R}) where {T,N,S,A<:Array{S},R<:Base.ReinterpretArray{T,N,S,A}}
    return ntuple(static, StaticInt(N))
end
@inline function stride_rank(::Type{A}) where {NB,NA,B<:AbstractArray{<:Any,NB},A<:Base.ReinterpretArray{<:Any,NA,<:Any,B,true}}
    NA == NB ? stride_rank(B) : _stride_rank_reinterpret(stride_rank(B), gt(StaticInt{NB}(), StaticInt{NA}()))
end
@inline function stride_rank(::Type{A}) where {N,B<:AbstractArray{<:Any,N},A<:Base.ReinterpretArray{<:Any,N,<:Any,B,false}}
    stride_rank(B)
end

@inline _stride_rank_reinterpret(sr, ::False) = (One(), map(Base.Fix2(+, One()), sr)...)
@inline _stride_rank_reinterpret(sr::Tuple{One,Vararg}, ::True) = map(Base.Fix2(-, One()), tail(sr))
function contiguous_axis(::Type{R}) where {T,N,S,B<:AbstractArray{S,N},R<:ReinterpretArray{T,N,S,B,false}}
    contiguous_axis(B)
end
# if the leading dim's `stride_rank` is not one, then that means the individual elements are split across an axis, which ArrayInterface
# doesn't currently have a means of representing.
@inline function contiguous_axis(::Type{A}) where {NB,NA,B<:AbstractArray{<:Any,NB},A<:Base.ReinterpretArray{<:Any,NA,<:Any,B,true}}
    _reinterpret_contiguous_axis(stride_rank(B), dense_dims(B), contiguous_axis(B), gt(StaticInt{NB}(), StaticInt{NA}()))
end
@inline _reinterpret_contiguous_axis(::Any, ::Any, ::Any, ::False) = One()
@inline _reinterpret_contiguous_axis(::Any, ::Any, ::Any, ::True) = Zero()
@generated function _reinterpret_contiguous_axis(t::Tuple{One,Vararg{StaticInt,N}}, d::Tuple{True,Vararg{StaticBool,N}}, ::One, ::True) where {N}
    for n in 1:N
        if t.parameters[n+1].parameters[1] === 2
            if d.parameters[n+1] === True
                return :(StaticInt{$n}())
            else
                return :(Zero())
            end
        end
    end
    :(Zero())
end

function stride_rank(::Type{Base.ReshapedArray{T, N, P, Tuple{Vararg{Base.SignedMultiplicativeInverse{Int},M}}}}) where {T,N,P,M}
    _reshaped_striderank(is_column_major(P), Val{N}(), Val{M}())
end
function stride_rank(::Type{Base.ReshapedArray{T, 1, A, Tuple{}}}) where {T, A}
    IfElse.ifelse(is_column_major(A) & is_dense(A), (static(1),), nothing)
end
function stride_rank(::Type{Base.ReshapedArray{T, 1, LinearAlgebra.Adjoint{T, A}, Tuple{}}}) where {T, A <: AbstractVector{T}}
    IfElse.ifelse(is_dense(A), (static(1),), nothing)
end
function stride_rank(::Type{Base.ReshapedArray{T, 1, LinearAlgebra.Transpose{T, A}, Tuple{}}}) where {T, A <: AbstractVector{T}}
    IfElse.ifelse(is_dense(A), (static(1),), nothing)
end

_reshaped_striderank(::True, ::Val{N}, ::Val{0}) where {N} = ntuple(static, StaticInt(N))
_reshaped_striderank(_, __, ___) = nothing

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
_contiguous_batch_size(::StaticInt{-1}, ::R) where {R<:Tuple} = -One()

contiguous_batch_size(::Type{Array{T,N}}) where {T,N} = Zero()
contiguous_batch_size(::Type{BitArray{N}}) where {N} = Zero()
contiguous_batch_size(@nospecialize T::Type{<:AbstractRange}) = Zero()
contiguous_batch_size(@nospecialize T::Type{<:Tuple}) = Zero()
@inline function contiguous_batch_size(@nospecialize T::Type{<:Union{PermutedDimsArray,Transpose,Adjoint}})
    contiguous_batch_size(parent_type(T))
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
    is_column_major(A) -> StaticBool

Returns `True()` if elements of `A` are stored in column major order. Otherwise returns `False()`.
"""
is_column_major(A) = is_column_major(stride_rank(A), contiguous_batch_size(A))
is_column_major(sr::Nothing, cbs) = False()
is_column_major(sr::R, cbs) where {R} = _is_column_major(sr, cbs)
is_column_major(::AbstractRange) = False()

# cbs > 0
_is_column_major(sr::R, cbs::StaticInt) where {R} = False()
# cbs <= 0
_is_column_major(sr::R, cbs::Union{StaticInt{0},StaticInt{-1}}) where {R} = is_increasing(sr)
function is_increasing(perm::Tuple{StaticInt{X},StaticInt{Y},Vararg}) where {X, Y}
    X <= Y ? is_increasing(tail(perm)) : False()
end
is_increasing(::Tuple{StaticInt{X},StaticInt{Y}}) where {X, Y} = X <= Y ? True() : False()
is_increasing(::Tuple{StaticInt{X}}) where {X} = True()
is_increasing(::Tuple{}) = True()

"""
    dense_dims(::Type{<:AbstractArray{N}}) -> Tuple{Vararg{StaticBool,N}}

Returns a tuple of indicators for whether each axis is dense.
An axis `i` of array `A` is dense if `stride(A, i) * Base.size(A, i) == stride(A, j)`
where `stride_rank(A)[i] + 1 == stride_rank(A)[j]`.
"""
dense_dims(x) = dense_dims(typeof(x))
function dense_dims(::Type{T}) where {T}
    is_forwarding_wrapper(T) ? dense_dims(parent_type(T)) : nothing
end
_all_dense(::Val{N}) where {N} = ntuple(_ -> True(), Val{N}())

function dense_dims(@nospecialize T::Type{<:DenseArray})
    ntuple(Compat.Returns(True()), StaticInt(ndims(T)))
end
dense_dims(::Type{BitArray{N}}) where {N} = ntuple(Compat.Returns(True()), StaticInt(N))
dense_dims(@nospecialize T::Type{<:AbstractRange}) = (True(),)
dense_dims(@nospecialize T::Type{<:Tuple}) = (True(),)
function dense_dims(@nospecialize T::Type{<:VecAdjTrans})
    dense = dense_dims(parent_type(T))
    if dense === nothing
        return nothing
    else
        return (True(), first(dense))
    end
end
@inline function dense_dims(@nospecialize T::Type{<:Union{MatAdjTrans,PermutedDimsArray}})
    dense = dense_dims(parent_type(T))
    if dense === nothing
        return nothing
    else
        return map(GetIndex{false}(dense), to_parent_dims(T))
    end
end

@inline function dense_dims(@nospecialize T::Type{<:Base.ReshapedReinterpretArray})
    ddb = dense_dims(parent_type(T))
    IfElse.ifelse(Static.le(StaticInt(ndims(parent_type(T))), StaticInt(ndims(T))), (True(), ddb...), Base.tail(ddb))
end
@inline dense_dims(@nospecialize T::Type{<:Base.NonReshapedReinterpretArray}) = dense_dims(parent_type(T))

@inline function dense_dims(@nospecialize T::Type{<:SubArray})
    dd = dense_dims(parent_type(T))
    if dd === nothing
        return nothing
    else
        return _dense_dims(T, dd, Val(stride_rank(parent_type(T))))
    end
end

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
    if static_length(dense_tup.args) === N
        return dense_tup
    else
        return nothing
    end
end

function dense_dims(@nospecialize T::Type{<:Base.ReshapedArray})
    d = dense_dims(parent_type(T))
    if d === nothing
        return nothing
    elseif all(d)
        return ntuple(Compat.Returns(True()), StaticInt(ndims(T)))
    else
        return ntuple(Compat.Returns(False()), StaticInt(ndims(T)))
    end
end

is_dense(A) = all(dense_dims(A)) ? True() : False()

"""
    known_strides(::Type{T}) -> Tuple
    known_strides(::Type{T}, dim) -> Union{Int,Nothing}

Returns the strides of array `A` known at compile time. Any strides that are not known at
compile time are represented by `nothing`.
"""
known_strides(x, dim) = known_strides(typeof(x), dim)
known_strides(::Type{T}, dim) where {T} = known_strides(T, to_dims(T, dim))
function known_strides(::Type{T}, dim::IntType) where {T}
    # see https://github.com/JuliaLang/julia/blob/6468dcb04ea2947f43a11f556da9a5588de512a0/base/reinterpretarray.jl#L148
    if ndims(T) < dim
        return known_length(T)
    else
        return known_strides(T)[dim]
    end
end
known_strides(::Type{<:StrideIndex{N,R,C,S,O}}) where {N,R,C,S,O} = known(S)

known_strides(x) = known_strides(typeof(x))
known_strides(::Type{T}) where {T<:Vector} = (1,)
@inline function known_strides(::Type{T}) where {T<:VecAdjTrans}
    strd = first(known_strides(parent_type(T)))
    return (strd, strd)
end
@inline function known_strides(@nospecialize T::Type{<:Union{MatAdjTrans,PermutedDimsArray}})
    map(GetIndex{false}(known_strides(parent_type(T))), to_parent_dims(T))
end
@inline function known_strides(::Type{T}) where {T<:SubArray}
    map(GetIndex{false}(known_strides(parent_type(T))), to_parent_dims(T))
end
function known_strides(::Type{T}) where {T}
    if ndims(T) === 1
        return (1,)
    else
        return size_to_strides(known_size(T), 1)
    end
end

"""
    static_strides(A) -> Tuple{Vararg{Union{Int,StaticInt}}}
    static_strides(A, dim) -> Union{Int,StaticInt}

Returns the strides of array `A`. If any strides are known at compile time,
these should be returned as `Static` numbers. For example:
```julia
julia> A = rand(3,4);

julia> ArrayInterface.static_strides(A)
(static(1), 3)
```

Additionally, the behavior differs from `Base.strides` for adjoint vectors:

```julia
julia> x = rand(5);

julia> ArrayInterface.static_strides(x')
(static(1), static(1))
```

This is to support the pattern of using just the first stride for linear indexing, `x[i]`,
while still producing correct behavior when using valid cartesian indices, such as `x[1,i]`.
```
"""
static_strides(A::StrideIndex) = getfield(A, :strides)
@inline static_strides(A::Vector{<:Any}) = (StaticInt(1),)
@inline static_strides(A::Array{<:Any,N}) where {N} = (StaticInt(1), Base.tail(Base.strides(A))...)
@inline function static_strides(x::X) where {X}
    if is_forwarding_wrapper(X)
        return static_strides(parent(x))
    elseif defines_strides(X)
        return size_to_strides(static_size(x), One())
    else
        return Base.strides(x)
    end
end

_is_column_dense(::A) where {A<:AbstractArray} =
    defines_strides(A) &&
    (ndims(A) == 0 || Bool(is_dense(A)) && Bool(is_column_major(A)))

# Fixes the example of https://github.com/JuliaArrays/ArrayInterface.jl/issues/160
function static_strides(A::ReshapedArray)
    if _is_column_dense(parent(A))
        return size_to_strides(static_size(A), One())
    else
        pst = static_strides(parent(A))
        psz = static_size(parent(A))
        # Try dimension merging in order (starting from dim1).
        # `sz1` and `st1` are the `size`/`stride` of dim1 after dimension merging.
        # `n` indicates the last merged dimension.
        # note: `st1` should be static if possible
        sz1, st1, n = merge_adjacent_dim(psz, pst)
        n == ndims(A.parent) && return size_to_strides(static_size(A), st1)
        return _reshaped_strides(static_size(A), One(), sz1, st1, n, Dims(psz), Dims(pst))
    end
end

@inline function _reshaped_strides(::Dims{0}, reshaped, msz::Int, _, ::Int, ::Dims, ::Dims)
    reshaped == msz && return ()
    throw(ArgumentError("Input is not strided."))
end
function _reshaped_strides(asz::Dims, reshaped, msz::Int, mst, n::Int, apsz::Dims, apst::Dims)
    st = reshaped * mst
    reshaped = reshaped * asz[1]
    if static_length(asz) > 1 && reshaped == msz && asz[2] != 1
        msz, mst′, n = merge_adjacent_dim(apsz, apst, n + 1)
        reshaped = 1
    else
        mst′ = Int(mst)
    end
    sts = _reshaped_strides(tail(asz), reshaped, msz, mst′, n, apsz, apst)
    return (st, sts...)
end

merge_adjacent_dim(::Tuple{}, ::Tuple{}) = 1, One(), 0
merge_adjacent_dim(szs::Tuple{Any}, sts::Tuple{Any}) = Int(szs[1]), sts[1], 1
function merge_adjacent_dim(szs::Tuple, sts::Tuple)
    if szs[1] isa One # Just ignore dimension with size 1
        sz, st, n = merge_adjacent_dim(tail(szs), tail(sts))
        return sz, st, n + 1
    elseif szs[2] isa One # Just ignore dimension with size 1
        sz, st, n = merge_adjacent_dim((szs[1], tail(tail(szs))...), (sts[1], tail(tail(sts))...))
        return sz, st, n + 1
    elseif (szs[1], szs[2], sts[1], sts[2]) isa NTuple{4,StaticInt} # the check could be done during compiling.
        if sts[2] == sts[1] * szs[1]
            szs′ = (szs[1] * szs[2], tail(tail(szs))...)
            sts′ = (sts[1], tail(tail(sts))...)
            sz, st, n = merge_adjacent_dim(szs′, sts′)
            return sz, st, n + 1
        else
            return Int(szs[1]), sts[1], 1
        end
    else # the check can't be done during compiling.
        sz, st, n = merge_adjacent_dim(Dims(szs), Dims(sts), 1)
        if (szs[1], sts[1]) isa NTuple{2,StaticInt} && szs[1] != 1
            # But the 1st stride might still be static.
            return sz, sts[1], n
        else
            return sz, st, n
        end
    end
end

function merge_adjacent_dim(psz::Dims{N}, pst::Dims{N}, n::Int) where {N}
    sz, st = psz[n], pst[n]
    while n < N
        szₙ, stₙ = psz[n+1], pst[n+1]
        if sz == 1
            sz, st = szₙ, stₙ
        elseif stₙ == st * sz
            sz *= szₙ
        elseif szₙ != 1
            break
        end
        n += 1
    end
    return sz, st, n
end

# `static_strides` for `Base.ReinterpretArray`
function static_strides(A::Base.ReinterpretArray{T,<:Any,S,<:AbstractArray{S},IsReshaped}) where {T,S,IsReshaped}
    _is_column_dense(parent(A)) && return size_to_strides(static_size(A), One())
    stp = static_strides(parent(A))
    ET, ES = static(sizeof(T)), static(sizeof(S))
    ET === ES && return stp
    IsReshaped && ET < ES && return (One(), _reinterp_strides(stp, ET, ES)...)
    first(stp) == 1 || throw(ArgumentError("Parent must be contiguous in the 1st dimension!"))
    if IsReshaped
        # The wrapper tell us `A`'s parent has static size in dim1.
        # We can make the next stride static if the following dim is still dense.
        sr = stride_rank(parent(A))
        dd = dense_dims(parent(A))
        stp′ = _new_static(stp, sr, dd, ET ÷ ES)
        return _reinterp_strides(tail(stp′), ET, ES)
    else
        return (One(), _reinterp_strides(tail(stp), ET, ES)...)
    end
end
_new_static(P,_,_,_) = P # This should never be called, just in case.
@generated function _new_static(p::P, ::SR, ::DD, ::StaticInt{S}) where {S,N,P<:NTuple{N,Union{Int,StaticInt}},SR<:NTuple{N,StaticInt},DD<:NTuple{N,StaticBool}}
    sr = fieldtypes(SR)
    j = findfirst(T -> T() == sr[1]()+1, sr)
    if !isnothing(j) && !(fieldtype(P, j) <: StaticInt) && fieldtype(DD, j) === True
        return :(tuple($((i == j ? :(static($S)) : :(p[$i]) for i in 1:N)...)))
    else
        return :(p)
    end
end
@inline function _reinterp_strides(stp::Tuple, els::StaticInt, elp::StaticInt)
    if elp % els == 0
        N = elp ÷ els
        return map(Base.Fix2(*, N), stp)
    else
        return map(stp) do i
            d, r = divrem(elp * i, els)
            iszero(r) || throw(ArgumentError("Parent's strides could not be exactly divided!"))
            d
        end
    end
end

static_strides(@nospecialize x::AbstractRange) = (One(),)
function static_strides(x::VecAdjTrans)
    st = first(static_strides(parent(x)))
    return (st, st)
end
@inline function static_strides(x::Union{MatAdjTrans,PermutedDimsArray})
    map(GetIndex{false}(static_strides(parent(x))), to_parent_dims(x))
end

getmul(x::Tuple, y::Tuple, ::StaticInt{i}) where {i} = getfield(x, i) * getfield(y, i)
function static_strides(A::SubArray)
    eachop(getmul, to_parent_dims(typeof(A)), map(maybe_static_step, A.indices), static_strides(parent(A)))
end

maybe_static_step(x::AbstractRange) = static_step(x)
maybe_static_step(_) = nothing

@generated function size_to_strides(sz::S, init) where {N,S<:Tuple{Vararg{Any,N}}}
    out = Expr(:block, Expr(:meta, :inline))
    t = Expr(:tuple, :init)
    prev = :init
    i = 1
    while i <= (N - 1)
        if S.parameters[i] <: Nothing || (i > 1 &&  t.args[i - 1] === :nothing)
            push!(t.args, :nothing)
        else
            next = Symbol(:val_, i)
            push!(out.args, :($next = $prev * getfield(sz, $i)))
            push!(t.args, next)
            prev = next
        end
        i += 1
    end
    push!(out.args, t)
    return out
end

static_strides(a, dim) = static_strides(a, to_dims(a, dim))
function static_strides(a::A, dim::IntType) where {A}
    if is_forwarding_wrapper(A)
        return static_strides(parent(a), dim)
    else
        return Base.stride(a, Int(dim))
    end
end

@inline static_stride(A::AbstractArray, ::StaticInt{N}) where {N} = static_strides(A)[N]
@inline static_stride(A::AbstractArray, ::Val{N}) where {N} = static_strides(A)[N]
static_stride(A, i) = Base.stride(A, i) # for type stability
