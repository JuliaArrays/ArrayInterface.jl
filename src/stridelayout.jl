
#=
    stride_preserving_index(::Type{T}) -> StaticBool

Returns `True` if strides between each element can still be derived when indexing with an
instance of type `T`.
=#
stride_preserving_index(::Type{T}) where {T<:AbstractRange} = True()
stride_preserving_index(::Type{T}) where {T<:Int} = True()
stride_preserving_index(::Type{T}) where {T} = False()
function stride_preserving_index(::Type{T}) where {N,T<:Tuple{Vararg{Any,N}}}
    if all(eachop(_stride_preserving_index, nstatic(Val(N)), T))
        return True()
    else
        return False()
    end
end
function _stride_preserving_index(::Type{T}, i::StaticInt) where {T}
    return stride_preserving_index(field_type(T, i))
end

"""
    known_offsets(::Type{T}) -> Tuple
    known_offsets(::Type{T}, dim) -> Union{Int,Nothing}

Returns a tuple of offset values known at compile time. If the offset of a given axis is
not known at compile time `nothing` is returned its position.
"""
known_offsets(x, dim) = known_offsets(typeof(x), dim)
known_offsets(::Type{T}, dim) where {T} = known_offsets(T, to_dims(T, dim))
function known_offsets(::Type{T}, dim::Integer) where {T}
    if ndims(T) < dim
        return 1
    else
        return known_offsets(T)[dim]
    end
end

known_offsets(x) = known_offsets(typeof(x))
function known_offsets(::Type{T}) where {T}
    return eachop(_known_offsets, nstatic(Val(ndims(T))), axes_types(T))
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
offsets(x) = eachop(_offsets, nstatic(Val(ndims(x))), x)
function _offsets(x::X, dim::StaticInt{D}) where {X,D}
    start = known_first(axes_types(X, dim))
    if start === nothing
        return first(axes(x, dim))
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
    if parent_type(T) <: T
        return nothing
    else
        return contiguous_axis(parent_type(T))
    end
end
contiguous_axis(::Type{<:DenseArray}) = One()
contiguous_axis(::Type{<:BitArray}) = One()
contiguous_axis(::Type{<:AbstractRange}) = One()
contiguous_axis(::Type{<:Tuple}) = One()
function contiguous_axis(::Type{T}) where {T<:VecAdjTrans}
    c = contiguous_axis(parent_type(T))
    if c === nothing
        return nothing
    elseif c === One()
        return StaticInt{2}()
    else
        return -One()
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
function contiguous_axis(::Type{T}) where {T<:PermutedDimsArray}
    c = contiguous_axis(parent_type(T))
    if c === nothing
        return nothing
    elseif isone(-c)
        return c
    else
        return from_parent_dims(T, c)
    end
end
function contiguous_axis(::Type{Base.ReshapedArray{T, 1, A, Tuple{}}}) where {T, A}
  IfElse.ifelse(is_column_major(A) & is_dense(A), static(1), nothing)
end
function contiguous_axis(::Type{Base.ReshapedArray{T, 1, LinearAlgebra.Adjoint{T, A}, Tuple{}}}) where {T, A <: AbstractVector{T}}
  IfElse.ifelse(is_column_major(A) & is_dense(A), static(1), nothing)
end
function contiguous_axis(::Type{Base.ReshapedArray{T, 1, LinearAlgebra.Transpose{T, A}, Tuple{}}}) where {T, A <: AbstractVector{T}}
  IfElse.ifelse(is_column_major(A) & is_dense(A), static(1), nothing)
end
function contiguous_axis(::Type{T}) where {T<:SubArray}
    return _contiguous_axis(T, contiguous_axis(parent_type(T)))
end

_contiguous_axis(::Type{A}, ::Nothing) where {T,N,P,I,A<:SubArray{T,N,P,I}} = nothing
_contiguous_axis(::Type{A}, c::StaticInt{-1}) where {T,N,P,I,A<:SubArray{T,N,P,I}} = c
function _contiguous_axis(::Type{A}, c::StaticInt{C}) where {T,N,P,I,A<:SubArray{T,N,P,I},C}
    if field_type(I, c) <: AbstractUnitRange
        return from_parent_dims(A)[C]
    elseif field_type(I, c) <: AbstractArray
        return -One()
    elseif field_type(I, c) <: Integer
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
    contiguous_axis_indicator(::Type{T}) -> Tuple{Vararg{StaticBool}}

Returns a tuple boolean `Val`s indicating whether that axis is contiguous.
"""
function contiguous_axis_indicator(::Type{A}) where {D,A<:AbstractArray{<:Any,D}}
    return contiguous_axis_indicator(contiguous_axis(A), Val(D))
end
contiguous_axis_indicator(::A) where {A<:AbstractArray} = contiguous_axis_indicator(A)
contiguous_axis_indicator(::Nothing, ::Val) = nothing
function contiguous_axis_indicator(c::StaticInt{N}, dim::Val{D}) where {N,D}
    return map(i -> eq(c, i), nstatic(dim))
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
    if parent_type(T) <: T
        return nothing
    else
        return stride_rank(parent_type(T))
    end
end
stride_rank(::Type{<:DenseArray{T,N}}) where {T,N} = nstatic(Val(N))
stride_rank(::Type{BitArray{N}}) where {N} = nstatic(Val(N))
stride_rank(::Type{<:AbstractRange}) = (One(),)
stride_rank(::Type{<:Tuple}) = (One(),)

stride_rank(::Type{T}) where {T<:VecAdjTrans} = (StaticInt(2), StaticInt(1))
stride_rank(::Type{T}) where {T<:MatAdjTrans} = _stride_rank(T, stride_rank(parent_type(T)))
_stride_rank(::Type{T}, ::Nothing) where {T<:MatAdjTrans} = nothing
function _stride_rank(::Type{T}, rank) where {T<:MatAdjTrans}
    return (getfield(rank, 2), getfield(rank, 1))
end

function stride_rank(::Type{T},) where {T<:PermutedDimsArray}
    return _stride_rank(T, stride_rank(parent_type(T)))
end
_stride_rank(::Type{T}, ::Nothing) where {T<:PermutedDimsArray} = nothing
_stride_rank(::Type{T}, r) where {T<:PermutedDimsArray} = permute(r, to_parent_dims(T))

stride_rank(::Type{T}) where {T<:SubArray} = _stride_rank(T, stride_rank(parent_type(T)))
_stride_rank(::Any, ::Any) = nothing
_stride_rank(::Type{T}, r::Tuple) where {T<:SubArray} = permute(r, to_parent_dims(T))

stride_rank(x, i) = stride_rank(x)[i]
function stride_rank(::Type{R}) where {T,N,S,A<:Array{S},R<:Base.ReinterpretArray{T,N,S,A}}
    return nstatic(Val(N))
end
if VERSION ≥ v"1.6.0-DEV.1581"
  @inline function stride_rank(::Type{A}) where {NB, NA, B <: AbstractArray{<:Any,NB},A<: Base.ReinterpretArray{<:Any, NA, <:Any, B, true}}
    NA == NB ? stride_rank(B) : _stride_rank_reinterpret(stride_rank(B), gt(StaticInt{NB}(), StaticInt{NA}()))
  end
  @inline _stride_rank_reinterpret(sr, ::False) = (One(), map(Base.Fix2(+,One()),sr)...)
  @inline _stride_rank_reinterpret(sr::Tuple{One,Vararg}, ::True) = map(Base.Fix2(-,One()), tail(sr))
  # if the leading dim's `stride_rank` is not one, then that means the individual elements are split across an axis, which ArrayInterface
  # doesn't currently have a means of representing.
  @inline function contiguous_axis(::Type{A}) where {NB, NA, B <: AbstractArray{<:Any,NB},A<: Base.ReinterpretArray{<:Any, NA, <:Any, B, true}}
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

_reshaped_striderank(::True, ::Val{N}, ::Val{0}) where {N} = nstatic(Val(N))
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
contiguous_batch_size(::Type{<:AbstractRange}) = Zero()
contiguous_batch_size(::Type{<:Tuple}) = Zero()
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

"""
    dense_dims(::Type{<:AbstractArray{N}}) -> Tuple{Vararg{StaticBool,N}}

Returns a tuple of indicators for whether each axis is dense.
An axis `i` of array `A` is dense if `stride(A, i) * Base.size(A, i) == stride(A, j)`
where `stride_rank(A)[i] + 1 == stride_rank(A)[j]`.
"""
dense_dims(x) = dense_dims(typeof(x))
function dense_dims(::Type{T}) where {T}
    if parent_type(T) <: T
        return nothing
    else
        return dense_dims(parent_type(T))
    end
end
_all_dense(::Val{N}) where {N} = ntuple(_ -> True(), Val{N}())

dense_dims(::Type{<:DenseArray{T,N}}) where {T,N} = _all_dense(Val{N}())
dense_dims(::Type{BitArray{N}}) where {N} = _all_dense(Val{N}())
dense_dims(::Type{<:AbstractRange}) = (True(),)
dense_dims(::Type{<:Tuple}) = (True(),)
function dense_dims(::Type{T}) where {T<:VecAdjTrans}
    dense = dense_dims(parent_type(T))
    if dense === nothing
        return nothing
    else
        return (True(), first(dense))
    end
end
function dense_dims(::Type{T}) where {T<:MatAdjTrans}
    dense = dense_dims(parent_type(T))
    if dense === nothing
        return nothing
    else
        return (last(dense), first(dense))
    end
end
function dense_dims(::Type{T}) where {T<:PermutedDimsArray}
    dense = dense_dims(parent_type(T))
    if dense === nothing
        return nothing
    else
        return permute(dense, to_parent_dims(T))
    end
end
function dense_dims(::Type{S}) where {N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}}
    return _dense_dims(S, dense_dims(A), Val(stride_rank(A)))
end
if VERSION ≥ v"1.6.0-DEV.1581"
    @inline function dense_dims(::Type{A}) where {NB, NA, B <: AbstractArray{<:Any,NB},A<: Base.ReinterpretArray{<:Any, NA, <:Any, B, true}}
        ddb = dense_dims(B)
        IfElse.ifelse(Static.le(StaticInt(NB), StaticInt(NA)), (True(), ddb...), Base.tail(ddb))
    end
end

_dense_dims(::Type{S}, ::Nothing, ::Val{R}) where {R,N,NP,T,A<:AbstractArray{T,NP},I,S<:SubArray{T,N,A,I}} = nothing
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
is_dense(A) = is_dense(typeof(A))
is_dense(::Type{A}) where {A} = _is_dense(dense_dims(A))
_is_dense(::Tuple{False,Vararg}) = False()
_is_dense(t::Tuple{True,Vararg}) = _is_dense(Base.tail(t))
_is_dense(t::Tuple{True}) = True()
_is_dense(t::Tuple{}) = True()


_reshaped_dense_dims(_, __, ___, ____) = nothing
function _reshaped_dense_dims(dense::D, ::True, ::Val{N}, ::Val{0}) where {D,N}
    if all(dense)
        return _all_dense(Val{N}())
    else
        return nothing
    end
end
function _reshaped_dense_dims(dense::Tuple{Static.False}, ::True, ::Val{N}, ::Val{0}) where {N}
    return return ntuple(_ -> False(), Val{N}())
end

"""
    known_strides(::Type{T}) -> Tuple
    known_strides(::Type{T}, dim) -> Union{Int,Nothing}

Returns the strides of array `A` known at compile time. Any strides that are not known at
compile time are represented by `nothing`.
"""
known_strides(x, dim) = known_strides(typeof(x), dim)
known_strides(::Type{T}, dim) where {T} = known_strides(T, to_dims(T, dim))
function known_strides(::Type{T}, dim::Integer) where {T}
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
function known_strides(::Type{T}) where {T<:MatAdjTrans}
    return permute(known_strides(parent_type(T)), to_parent_dims(T))
end
@inline function known_strides(::Type{T}) where {T<:VecAdjTrans}
    strd = first(known_strides(parent_type(T)))
    return (strd, strd)
end
@inline function known_strides(::Type{T}) where {T<:PermutedDimsArray}
    return permute(known_strides(parent_type(T)), to_parent_dims(T))
end
@inline function known_strides(::Type{T}) where {T<:SubArray}
    return permute(known_strides(parent_type(T)), to_parent_dims(T))
end
function known_strides(::Type{T}) where {T}
    if ndims(T) === 1
        return (1,)
    else
        return size_to_strides(known_size(T), 1)
    end
end

"""
    strides(A) -> Tuple{Vararg{Union{Int,StaticInt}}}
    strides(A, dim) -> Union{Int,StaticInt}

Returns the strides of array `A`. If any strides are known at compile time,
these should be returned as `Static` numbers. For example:
```julia
julia> A = rand(3,4);

julia> ArrayInterface.strides(A)
(static(1), 3)
```

Additionally, the behavior differs from `Base.strides` for adjoint vectors:

```julia
julia> x = rand(5);

julia> ArrayInterface.strides(x')
(static(1), static(1))
```

This is to support the pattern of using just the first stride for linear indexing, `x[i]`,
while still producing correct behavior when using valid cartesian indices, such as `x[1,i]`.
```
"""
strides(A::StrideIndex) = getfield(A, :strides)
@inline strides(A::Vector{<:Any}) = (StaticInt(1),)
@inline strides(A::Array{<:Any,N}) where {N} = (StaticInt(1), Base.tail(Base.strides(A))...)
@inline function strides(x::X) where {X}
    if !(parent_type(X) <: X)
        return strides(parent(x))
    elseif defines_strides(X)
        return size_to_strides(size(x), One())
    else
        return Base.strides(x)
    end
end

# Fixes the example of https://github.com/JuliaArrays/ArrayInterface.jl/issues/160
# TODO: Should be generalized to reshaped arrays wrapping more general array types
function strides(A::ReshapedArray{T,N,P}) where {T, N, P<:AbstractVector}
    if defines_strides(A)
        return size_to_strides(size(A), first(strides(parent(A))))
    else
        return Base.strides(A)
    end
end
function strides(A::ReshapedArray{T,N,P}) where {T, N, P}
    if defines_strides(A)
        return size_to_strides(size(A), static(1))
    else
        return Base.strides(A)
    end
end


@inline bmap(f::F, t::Tuple{}, x::Number) where {F} = ()
@inline bmap(f::F, t::Tuple{T}, x::Number) where {F, T} = (f(first(t),x), )
@inline bmap(f::F, t::Tuple, x::Number) where {F} = (f(first(t),x), bmap(f, Base.tail(t), x)...)
@static if VERSION ≥ v"1.6.0-DEV.1581"
  # from `reinterpret(reshape, ...)`
  @inline function strides(A::Base.ReinterpretArray{R, N, T, B, true}) where {R,N,T,B}
    P = strides(parent(A))
    if sizeof(R) == sizeof(T)
      P
    elseif sizeof(R) > sizeof(T)
      x = Base.tail(P)
      fx = first(x)
      if fx isa Int
        (One(), bmap(Base.sdiv_int, Base.tail(x), fx)...)
      else
        (One(), bmap(÷, Base.tail(x), fx)...)
      end
    else
      (One(), bmap(*, P, StaticInt(sizeof(T)) ÷ StaticInt(sizeof(R)))...)
    end
  end
  # plain `reinterpret(...)`
  @inline function strides(A::Base.ReinterpretArray{R, N, T, B, false}) where {R,N,T,B}
    P = strides(parent(A))
    if sizeof(R) == sizeof(T)
      P
    elseif sizeof(R) > sizeof(T)
      (first(P), bmap(÷, Base.tail(P), StaticInt(sizeof(R)) ÷ StaticInt(sizeof(T)))...)
    else # sizeof(R) < sizeof(T)
      (first(P), bmap(*, Base.tail(P), StaticInt(sizeof(T)) ÷ StaticInt(sizeof(R)))...)
    end
  end
else
  # plain `reinterpret(...)`
  @inline function strides(A::Base.ReinterpretArray{R, N, T}) where {R,N,T}
    P = strides(parent(A))
    if sizeof(R) == sizeof(T)
      P
    elseif sizeof(R) > sizeof(T)
      (first(P), bmap(÷, Base.tail(P), StaticInt(sizeof(R)) ÷ StaticInt(sizeof(T)))...)
    else # sizeof(R) < sizeof(T)
      (first(P), bmap(*, Base.tail(P), StaticInt(sizeof(T)) ÷ StaticInt(sizeof(R)))...)
    end
  end
end
#@inline strides(A) = _strides(A, Base.strides(A), contiguous_axis(A))

strides(::AbstractRange) = (One(),)
function strides(x::VecAdjTrans)
    st = first(strides(parent(x)))
    return (st, st)
end
@inline strides(B::MatAdjTrans) = permute(strides(parent(B)), to_parent_dims(B))
@inline strides(B::PermutedDimsArray) = permute(strides(parent(B)), to_parent_dims(B))

getmul(x::Tuple, y::Tuple, ::StaticInt{i}) where {i} = getfield(x, i) * getfield(y, i)
function strides(A::SubArray)
    return eachop(getmul, to_parent_dims(A), map(maybe_static_step, A.indices), strides(parent(A)))
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

strides(a, dim) = strides(a, to_dims(a, dim))
function strides(a::A, dim::Integer) where {A}
    if parent_type(A) <: A
        return Base.stride(a, Int(dim))
    else
        return strides(parent(a), to_parent_dims(A, dim))
    end
end

@inline stride(A::AbstractArray, ::StaticInt{N}) where {N} = strides(A)[N]
@inline stride(A::AbstractArray, ::Val{N}) where {N} = strides(A)[N]
stride(A, i) = Base.stride(A, i) # for type stability

