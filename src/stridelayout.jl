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
contiguous_axis(::Type) = nothing
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
    new_contig = I2[_get(c)]
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
        if I.parameters[np] <: AbstractUnitRange
            n += 1
            if np == contig
                new_contig = n
            end
        else
            if np == contig
                new_contig = -1
            end
        end
    end
    # If n != N, then an axis was indeced by something other than an integer or `AbstractUnitRange`, so we return `nothing`
    n == N || return nothing
    Expr(:call, Expr(:curly, :Contiguous, new_contig))
end

"""
contiguous_axis_indicator(::Type{T}) -> Tuple{Vararg{<:Val}}

Returns a tuple boolean `Val`s indicating whether that axis is contiguous.
"""
contiguous_axis_indicator(::Type{A}) where {D, A <: AbstractArray{<:Any,D}} = contiguous_axis_indicator(contiguous_axis(A), Val(D))
contiguous_axis_indicator(::A) where {A <: AbstractArray} = contiguous_axis_indicator(A)
Base.@pure contiguous_axis_indicator(::Contiguous{N}, ::Val{D}) where {N,D} = ntuple(d -> Val{d == N}(), Val{D}())

"""
If the contiguous dimension is not the dimension with `Stride_rank{1}`
"""
struct ContiguousBatch{N} end
Base.@pure ContiguousBatch(N::Int) = ContiguousBatch{N}()
_get(::ContiguousBatch{N}) where {N} = N

"""
contiguous_batch_size(::Type{T}) -> ContiguousBatch{N}

Returns the size of contiguous batches if `!isone(stride_rank(T, contiguous_axis(T)))`.
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
        if I.parameters[np] <: AbstractUnitRange
            n += 1
            push!(rank_new, r)
        end
    end
    # If n != N, then an axis was indeced by something other than an integer or `AbstractUnitRange`, so we return `nothing`
    n == N || return nothing
    ranktup = Expr(:tuple); append!(ranktup.args, rank_new) # dynamic splats bad
    Expr(:call, Expr(:curly, :StrideRank, ranktup))
end
stride_rank(x, i) = stride_rank(x)[i]

struct DenseDims{D} end
Base.@pure DenseDims(D::NTuple{<:Any,Bool}) = DenseDims{D}()
@inline Base.getindex(::DenseDims{D}, i::Integer) where {D} = D[i]
@inline Base.getindex(::DenseDims{D}, ::Val{I}) where {D,I} = DenseDims{permute(D, I)}()
"""
dense_dims(::Type{T}) -> NTuple{N,Bool}

Returns a tuple of indicators for whether each axis is dense.
An axis `i` of array `A` is dense if `stride(A, i) * size(A, i) == stride(A, j)` where `stride_rank(A)[i] + 1 == stride_rank(A)[j]`.
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
        spₙ = sp[np]
        if I.parameters[np] <: Base.Slice
            push!(dense_tup.args, densev[np])
        elseif I.parameters[np] <: AbstractUnitRange
            push!(dense_tup.args, densev[np])
        end
    end
    # If n != N, then an axis was indexed by something other than an integer or `AbstractUnitRange`, so we return `nothing`
    length(dense_tup.args) == N ? Expr(:call, Expr(:curly, :DenseDims, dense_tup)) : nothing
end

permute(t::NTuple{N}, I::NTuple{N,Int}) where {N} = ntuple(n -> t[I[n]], Val{N}())
@generated function permute(t::Tuple{Vararg{Any,N}}, ::Val{I}) where {N,I}
    t = Expr(:tuple)
    foreach(i -> push!(t.args, Expr(:ref, :t, i)), I)
    Expr(:block, Expr(:meta, :inline), t)
end

"""
  sdsize(A)

Returns the size of `A`. If the size of any axes are known at compile time,
these should be returned as `Static` numbers. For example:
```julia
julia> using StaticArrays; using ArrayInterface: sdsize

julia> A = @SMatrix rand(3,4);

julia> sdsize(A)
(Static{3}(), Static{4}())
```
"""
sdsize(::Any) = nothing
"""
  sdstrides(A)

Returns the strides of array `A`. If any strides are known at compile time,
these should be returned as `Static` numbers. For example:
```julia
julia> using ArrayInterface: sdstrides

julia> A = rand(3,4);

julia> sdstrides(A)
(Static{1}(), 3)
```
"""
sdstrides(::Any) = nothing
"""
  sdoffsets(A)

Returns offsets of indices with respect to 0. If values are known at compile time,
it should return them as `Static` numbers.
For example, if `A isa Base.Matrix`, `sdoffsets(A) === (Static(1), Static(1))`.
"""
sdoffsets(::Any) = (Static{1}(),) # Assume arbitrary Julia data structures use 1-based indexing by default.
@inline sdsize(A::AbstractArray{<:Any,N}) where {N} = size(A)
@inline sdstrides(A::Vector{<:Any}) = (Static(1),)
@inline sdstrides(A::Array{<:Any,N}) where {N} = (Static(1), Base.tail(strides(A))...)
@inline sdstrides(A::AbstractArray{<:Any,N}) where {N} = strides(A)

@inline function sdoffsets(x, i)
    inds = indices(x, i)
    start = known_first(inds)
    isnothing(start) ? first(inds) : Static(start)
end
# @inline sdoffsets(A::AbstractArray{<:Any,N}) where {N} = ntuple(n -> sdoffsets(A, n), Val{N}())
# Explicit tuple needed for inference.
@generated function sdoffsets(A::AbstractArray{<:Any,N}) where {N}
    quote
        $(Expr(:meta, :inline))
        Base.Cartesian.@ntuple $N n -> sdoffsets(A, n)
    end
end


@inline sdsize(B::Union{Transpose{T,A},Adjoint{T,A}}) where {T,A<:AbstractMatrix{T}} = permute(sdsize(parent(B)), Val{(2,1)}())
@inline sdsize(B::PermutedDimsArray{T,N,I1,I2,A}) where {T,N,I1,I2,A<:AbstractArray{T,N}} = permute(sdsize(parent(B)), Val{I1}())
@inline sdstrides(B::Union{Transpose{T,A},Adjoint{T,A}}) where {T,A<:AbstractMatrix{T}} = permute(sdstrides(parent(B)), Val{(2,1)}())
@inline sdstrides(B::PermutedDimsArray{T,N,I1,I2,A}) where {T,N,I1,I2,A<:AbstractArray{T,N}} = permute(sdstrides(parent(B)), Val{I1}())

sdsize(B::S) where {N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}} = _sdsize(sdsize(parent(B)), B.indices, map(static_length, B.indices))
sdstrides(B::S) where {N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}} = _sdstrides(sdstrides(parent(B)), B.indices)
@generated function _sdsize(A::Tuple{Vararg{Any,N}}, inds::I, l::L) where {N, I<:Tuple, L}
    t = Expr(:tuple)
    for n in 1:N
        if (I.parameters[n] <: Base.Slice)
            push!(t.args, :(@inbounds(_try_static(A[$n], l[$n]))))
        elseif I.parameters[n] <: AbstractUnitRange
            push!(t.args, Expr(:ref, :l, n))
        end
    end
    Expr(:block, Expr(:meta, :inline), t)
end
@generated function _sdstrides(A::Tuple{Vararg{Any,N}}, inds::I) where {N, I<:Tuple}
    t = Expr(:tuple)
    for n in 1:N
        if I.parameters[n] <: AbstractRange
            push!(t.args, Expr(:ref, :A, n))
        elseif !(I.parameters[n] <: Integer)
            return nothing
        end
    end
    Expr(:block, Expr(:meta, :inline), t)
end

