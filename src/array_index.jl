
"""
    ArrayIndex{N}

Subtypes of `ArrayIndex` represent series of transformations for a provided index to some
buffer which is typically accomplished with square brackets (e.g., `buffer[index[inds...]]`).
The only behavior that is required of a subtype of `ArrayIndex` is the ability to transform
individual index elements (i.e. not collections). This does not guarantee bounds checking or
the ability to iterate (although additional functionallity may be provided for specific
types).
"""
abstract type ArrayIndex{N} end

const MatrixIndex = ArrayIndex{2}

const VectorIndex = ArrayIndex{1}

Base.ndims(::ArrayIndex{N}) where {N} = N
Base.ndims(::Type{<:ArrayIndex{N}}) where {N} = N

struct BidiagonalIndex <: MatrixIndex
    count::Int
    isup::Bool
end

struct TridiagonalIndex <: MatrixIndex
    count::Int# count==nsize+nsize-1+nsize-1
    nsize::Int
    isrow::Bool
end

struct BandedMatrixIndex <: MatrixIndex
    count::Int
    rowsize::Int
    colsize::Int
    bandinds::Array{Int,1}
    bandsizes::Array{Int,1}
    isrow::Bool
end

function _bandsize(bandind, rowsize, colsize)
    -(rowsize - 1) <= bandind <= colsize - 1 || throw(ErrorException("Invalid Bandind"))
    if (bandind * (colsize - rowsize) > 0) & (abs(bandind) <= abs(colsize - rowsize))
        return min(rowsize, colsize)
    elseif bandind * (colsize - rowsize) <= 0
        return min(rowsize, colsize) - abs(bandind)
    else
        return min(rowsize, colsize) - abs(bandind) + abs(colsize - rowsize)
    end
end

function BandedMatrixIndex(rowsize, colsize, lowerbandwidth, upperbandwidth, isrow)
    upperbandwidth > -lowerbandwidth || throw(ErrorException("Invalid Bandwidths"))
    bandinds = upperbandwidth:-1:-lowerbandwidth
    bandsizes = [_bandsize(band, rowsize, colsize) for band in bandinds]
    BandedMatrixIndex(sum(bandsizes), rowsize, colsize, bandinds, bandsizes, isrow)
end

struct BlockBandedMatrixIndex <: MatrixIndex
    count::Int
    refinds::Array{Int,1}
    refcoords::Array{Int,1}# storing col or row inds at ref points
    isrow::Bool
end

function BlockBandedMatrixIndex(nrowblock, ncolblock, rowsizes, colsizes, l, u)
    blockrowind = BandedMatrixIndex(nrowblock, ncolblock, l, u, true)
    blockcolind = BandedMatrixIndex(nrowblock, ncolblock, l, u, false)
    sortedinds = sort(
        [(blockrowind[i], blockcolind[i]) for i = 1:length(blockrowind)],
        by = x -> x[1],
    )
    sort!(sortedinds, by = x -> x[2], alg = InsertionSort)# stable sort keeps the second index in order
    refinds = Array{Int,1}()
    refrowcoords = Array{Int,1}()
    refcolcoords = Array{Int,1}()
    rowheights = pushfirst!(copy(rowsizes), 1)
    cumsum!(rowheights, rowheights)
    blockheight = 0
    blockrow = 1
    blockcol = 1
    currenti = 1
    lastrowind = sortedinds[1][1] - 1
    lastcolind = sortedinds[1][2]
    for ind in sortedinds
        rowind, colind = ind
        if colind == lastcolind
            if rowind > lastrowind
                blockheight += rowsizes[rowind]
            end
        else
            for j = blockcol:blockcol+colsizes[lastcolind]-1
                push!(refinds, currenti)
                push!(refrowcoords, blockrow)
                push!(refcolcoords, j)
                currenti += blockheight
            end
            blockcol += colsizes[lastcolind]
            blockrow = rowheights[rowind]
            blockheight = rowsizes[rowind]
        end
        lastcolind = colind
        lastrowind = rowind
    end
    for j = blockcol:blockcol+colsizes[lastcolind]-1
        push!(refinds, currenti)
        push!(refrowcoords, blockrow)
        push!(refcolcoords, j)
        currenti += blockheight
    end
    push!(refinds, currenti)# guard
    push!(refrowcoords, -1)
    push!(refcolcoords, -1)
    rowindobj = BlockBandedMatrixIndex(currenti - 1, refinds, refrowcoords, true)
    colindobj = BlockBandedMatrixIndex(currenti - 1, refinds, refcolcoords, false)
    rowindobj, colindobj
end

struct BandedBlockBandedMatrixIndex <: MatrixIndex
    count::Int
    refinds::Array{Int,1}
    refcoords::Array{Int,1}# storing col or row inds at ref points
    reflocalinds::Array{BandedMatrixIndex,1}
    isrow::Bool
end

function BandedBlockBandedMatrixIndex(
    nrowblock,
    ncolblock,
    rowsizes,
    colsizes,
    l,
    u,
    lambda,
    mu,
)
    blockrowind = BandedMatrixIndex(nrowblock, ncolblock, l, u, true)
    blockcolind = BandedMatrixIndex(nrowblock, ncolblock, l, u, false)
    sortedinds = sort(
        [(blockrowind[i], blockcolind[i]) for i = 1:length(blockrowind)],
        by = x -> x[1],
    )
    sort!(sortedinds, by = x -> x[2], alg = InsertionSort)# stable sort keeps the second index in order
    rowheights = pushfirst!(copy(rowsizes), 1)
    cumsum!(rowheights, rowheights)
    colwidths = pushfirst!(copy(colsizes), 1)
    cumsum!(colwidths, colwidths)
    currenti = 1
    refinds = Array{Int,1}()
    refrowcoords = Array{Int,1}()
    refcolcoords = Array{Int,1}()
    reflocalrowinds = Array{BandedMatrixIndex,1}()
    reflocalcolinds = Array{BandedMatrixIndex,1}()
    for ind in sortedinds
        rowind, colind = ind
        localrowind =
            BandedMatrixIndex(rowsizes[rowind], colsizes[colind], lambda, mu, true)
        localcolind =
            BandedMatrixIndex(rowsizes[rowind], colsizes[colind], lambda, mu, false)
        push!(refinds, currenti)
        push!(refrowcoords, rowheights[rowind])
        push!(refcolcoords, colwidths[colind])
        push!(reflocalrowinds, localrowind)
        push!(reflocalcolinds, localcolind)
        currenti += localrowind.count
    end
    push!(refinds, currenti)
    push!(refrowcoords, -1)
    push!(refcolcoords, -1)
    rowindobj = BandedBlockBandedMatrixIndex(
        currenti - 1,
        refinds,
        refrowcoords,
        reflocalrowinds,
        true,
    )
    colindobj = BandedBlockBandedMatrixIndex(
        currenti - 1,
        refinds,
        refcolcoords,
        reflocalcolinds,
        false,
    )
    rowindobj, colindobj
end

"""
    StrideIndex(x)

Subtype of `ArrayIndex` that transforms and index using stride layout information
derived from `x`.
"""
struct StrideIndex{N,R,C,S,O} <: ArrayIndex{N}
    strides::S
    offsets::O

    function StrideIndex{N,R,C}(s::S, o::O) where {N,R,C,S,O}
        return new{N,R::NTuple{N,Int},C,S,O}(s, o)
    end
    function StrideIndex{N,R,C}(a::A) where {N,R,C,A}
        return StrideIndex{N,R,C}(strides(a), offsets(a))
    end
    function StrideIndex(a::A) where {A}
        return StrideIndex{ndims(A),known(stride_rank(A)), known(contiguous_axis(A))}(a)
    end
end

"""
    PermutedIndex

Subtypes of `ArrayIndex` that is responsible for permuting each index prior to accessing
parent indices.
"""
struct PermutedIndex{N,I1,I2} <: ArrayIndex{N}
    PermutedIndex{N,I1,I2}() where {N,I1,I2} = new{N,I1,I2}()
    function PermutedIndex(p::Tuple{Vararg{StaticInt,N}}, ip::Tuple{Vararg{StaticInt}}) where {N}
        PermutedIndex{N,known(p),known(ip)}()
    end
end

"""
    SubIndex(indices)

Subtype of `ArrayIndex` that provides a multidimensional view of another `ArrayIndex`.
"""
struct SubIndex{N,I} <: ArrayIndex{N}
    indices::I

    SubIndex{N}(inds::Tuple) where {N} = new{N,typeof(inds)}(Base.ensure_indexable(inds))
    SubIndex(x::SubArray{T,N,P,I}) where {T,N,P,I} = new{N,I}(getfield(x, :indices))
end

"""
    LinearSubIndex(offset, stride)

Subtype of `ArrayIndex` that provides linear indexing for `Base.FastSubArray` and
`FastContiguousSubArray`.
"""
struct LinearSubIndex{O<:CanonicalInt,S<:CanonicalInt} <: VectorIndex
    offset::O
    stride::S
end

offset1(x::LinearSubIndex) = getfield(x, :offset)
stride1(x::LinearSubIndex) = getfield(x, :stride)

const OffsetIndex{O} = LinearSubIndex{O,StaticInt{1}}
OffsetIndex(offset::CanonicalInt) = LinearSubIndex(offset, static(1))

"""
    IdentityIndex{N}

Used to specify that indices don't need any transformation.
"""
struct IdentityIndex{N} <: ArrayIndex{N} end

"""
    UnkownIndex{N}

This default return type when calling `ArrayIndex{N}(x)`.
"""
struct UnkownIndex{N} <: ArrayIndex{N} end

"""
    ComposedIndex(outer, inner)

A subtype of `ArrayIndex` that lazily combines index `outer` and `inner`. Indexing a
`ComposedIndex` whith `i` is equivalent to `outer[inner[i]]`.
"""
struct ComposedIndex{N,O,I} <: ArrayIndex{N}
    outer::O
    inner::I

    ComposedIndex(i1::I1, i2::I2) where {I1,I2} = new{ndims(I1),I1,I2}(i1, i2)
end

outer(x::ComposedIndex) = getfield(x, :outer)
inner(x::ComposedIndex) = getfield(x, :inner)

@inline _to_cartesian(x) = CartesianIndices(indices(x, ntuple(+, Val(ndims(x)))))
@inline function _to_linear(x)
    N = ndims(x)
    StrideIndex{N,ntuple(+, Val(N)),nothing}(size_to_strides(size(x), static(1)), offsets(x))
end

"""
    ArrayIndex{N}(A)

Constructs a subtype of `ArrayIndex` such that an `N` dimensional indexing argument may be
converted to an appropriate state for accessing the buffer of `A`. For example:

```julia
julia> A = reshape(1:20, 4, 5);

julia> index = ArrayInterface.ArrayIndex{2}(A);

julia> ArrayInterface.buffer(A)[index[2, 2]] == A[2, 2]
true

```
"""
ArrayIndex{N}(x) where {N} = UnkownIndex{N}()
ArrayIndex{N}(x::Array) where {N} = StrideIndex(x)
ArrayIndex{1}(x::Array) = OffsetIndex(static(0))

ArrayIndex{1}(x::ReshapedArray) = IdentityIndex{1}()
ArrayIndex{N}(x::ReshapedArray) where {N} = _to_linear(x)

# TODO should we only define index constructors for explicit types?
ArrayIndex{1}(x::AbstractRange) = OffsetIndex(offset1(x) - static(1))

## SubArray
ArrayIndex{N}(x::SubArray) where {N} = SubIndex{ndims(x)}(getfield(x, :indices))
@inline function ArrayIndex{1}(x::SubArray{T,N}) where {T,N}
    if N === 1
        return SubIndex(x)
    else
        return compose(SubIndex(x), _to_cartesian(x))
    end
end
ArrayIndex{1}(x::Base.FastContiguousSubArray) = OffsetIndex(getfield(x, :offset1))
function ArrayIndex{1}(x::Base.FastSubArray)
    LinearSubIndex(getfield(x, :offset1), getfield(x, :stride1))
end

## PermutedDimsArray
@inline function ArrayIndex{1}(x::PermutedDimsArray{T,N,I1,I2}) where {T,N,I1,I2}
    if N === 1
        return IdentityIndex{1}()
    else
        return compose(PermutedIndex{N,I1,I2}(), _to_cartesian(x))
    end
end
@inline ArrayIndex{N}(x::PermutedDimsArray{T,N,I1,I2}) where {T,N,I1,I2} = PermutedIndex{N,I1,I2}()

## Transpose/Adjoint{Real}
@inline function ArrayIndex{2}(x::Union{Transpose{<:Any,<:AbstractMatrix},Adjoint{<:Real,<:AbstractMatrix}})
    PermutedIndex{2,(2,1),(2,1)}()
end
@inline function ArrayIndex{2}(x::Union{Transpose{<:Any,<:AbstractVector},Adjoint{<:Real,<:AbstractVector}})
    PermutedIndex{2,(2,1),(2,)}()
end
@inline function ArrayIndex{1}(x::Union{Transpose{<:Any,<:AbstractMatrix},Adjoint{<:Real,<:AbstractMatrix}})
    compose(PermutedIndex{2,(2,1),(2,1)}(), _to_cartesian(x))
end
@inline function ArrayIndex{1}(x::Union{Transpose{<:Any,<:AbstractVector},Adjoint{<:Real,<:AbstractVector}})
    IdentityIndex{1}()
end

## Traits
Base.firstindex(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = 1
Base.lastindex(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = i.count
Base.length(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = i.count

## getindex
@propagate_inbounds Base.getindex(x::ArrayIndex, i::CanonicalInt, ii::CanonicalInt...) = x[NDIndex(i, ii...)]
Base.getindex(x::IdentityIndex, i::CanonicalInt) = 1
Base.getindex(x::IdentityIndex, i::AbstractCartesianIndex) = i
# we should be able to assume that if `i1` was indexed without error than it's inbounds
@propagate_inbounds Base.getindex(x::ComposedIndex) = @inbounds(outer(x)[inner(x)[]])
@propagate_inbounds Base.getindex(x::ComposedIndex, i::CanonicalInt) = @inbounds(outer(x)[inner(x)[i]])
@propagate_inbounds Base.getindex(x::ComposedIndex, i::AbstractCartesianIndex) = @inbounds(outer(x)[inner(x)[i]])

@propagate_inbounds function Base.getindex(ind::TridiagonalIndex, i::Int)
    @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
    offsetu = ind.isrow ? 0 : 1
    offsetl = ind.isrow ? 1 : 0
    if 1 <= i <= ind.nsize
        return i
    elseif ind.nsize < i <= ind.nsize + ind.nsize - 1
        return i - ind.nsize + offsetu
    else
        return i - (ind.nsize + ind.nsize - 1) + offsetl
    end
end

@propagate_inbounds function Base.getindex(ind::BandedMatrixIndex, i::Int)
    @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
    _i = i
    p = 1
    while _i - ind.bandsizes[p] > 0
        _i -= ind.bandsizes[p]
        p += 1
    end
    bandind = ind.bandinds[p]
    startfromone = !xor(ind.isrow, (bandind > 0))
    if startfromone
        return _i
    else
        return _i + abs(bandind)
    end
end

@propagate_inbounds function Base.getindex(ind::BlockBandedMatrixIndex, i::Int)
    @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
    p = 1
    while i - ind.refinds[p] >= 0
        p += 1
    end
    p -= 1
    _i = i - ind.refinds[p]
    if ind.isrow
        return ind.refcoords[p] + _i
    else
        return ind.refcoords[p]
    end
end

@propagate_inbounds function Base.getindex(ind::BandedBlockBandedMatrixIndex, i::Int)
    @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
    p = 1
    while i - ind.refinds[p] >= 0
        p += 1
    end
    p -= 1
    _i = i - ind.refinds[p] + 1
    ind.reflocalinds[p][_i] + ind.refcoords[p] - 1
end

@inline function Base.getindex(x::StrideIndex{N}, i::AbstractCartesianIndex{N}) where {N}
    return _strides2int(offsets(x), strides(x), Tuple(i)) + static(1)
end
@generated function _strides2int(o::O, s::S, i::I) where {O,S,I}
    N = known_length(S)
    out = :()
    for i in 1:N
        tmp = :(((getfield(i, $i) - getfield(o, $i)) * getfield(s, $i)))
        out = ifelse(i === 1, tmp, :($out + $tmp))
    end
    return Expr(:block, Expr(:meta, :inline), out)
end
function Base.getindex(x::PermutedIndex{2,(2,1),(2,)}, i::AbstractCartesianIndex{2})
    getfield(Tuple(i), 2)
end
@inline function Base.getindex(x::PermutedIndex{N,I1,I2}, i::AbstractCartesianIndex{N}) where {N,I1,I2}
    return NDIndex(permute(Tuple(i), Val(I2)))
end
@inline function Base.getindex(x::SubIndex{N}, i::AbstractCartesianIndex{N}) where {N}
    return NDIndex(Base.reindex(getfield(x, :indices), Tuple(i)))
end
@inline Base.getindex(x::LinearSubIndex, i::CanonicalInt) = offset1(x) + stride1(x) * i
@propagate_inbounds function Base.getindex(ind::BidiagonalIndex, i::Int)
    @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
    if ind.isup
        ii = i + 1
    else
        ii = i + 1 + 1
    end
    convert(Int, floor(ii / 2))
end

const compose = ∘

"""
    compose(outer_index, inner_index)
    outer_index ∘ inner_index

Given two subtypes of `ArrayIndex`, combines a new instance that when indexed is equivalent
to `i1[i2[i]]`. Default behavior produces a `ComposedIndex`, but more `i1` and `i2` may be
consolidated into a more efficient representation.
"""
compose(x::ArrayIndex, y::ArrayIndex) = _compose(x, y)
_compose(x, y::IdentityIndex) = x
_compose(x, y) = ComposedIndex(x, y)
_compose(x, y::ComposedIndex) = ComposedIndex(compose(x, outer(y)), inner(y))

compose(::IdentityIndex, y::ArrayIndex) = y
@inline compose(x::ComposedIndex, y::ArrayIndex) = ComposedIndex(outer(x), compose(inner(x), y))
@inline function compose(x::ComposedIndex, y::ComposedIndex)
    ComposedIndex(outer(x), ComposedIndex(compose(inner(x), outer(y)), inner(y)))
end
@inline function compose(x::StrideIndex, y::SubIndex{N,I}) where {N,I}
    _combined_sub_strides(stride_preserving_index(I), x, y)
end
_combined_sub_strides(::False, x::StrideIndex, i::SubIndex) = ComposedIndex(x, i)
@inline function _combined_sub_strides(::True, x::StrideIndex{N,R,C}, i::SubIndex{Ns,I}) where {N,R,C,Ns,I<:Tuple{Vararg{Any,N}}}
    c = static(C)
    if _get_tuple(I, c) <: AbstractUnitRange
        c2 = known(getfield(_from_sub_dims(I), C))
    elseif (_get_tuple(I, c) <: AbstractArray) && (_get_tuple(I, c) <: Integer)
        c2 = -1
    else
        c2 = nothing
    end

    pdims = _to_sub_dims(I)
    o = offsets(x)
    s = strides(x)
    inds = getfield(i, :indices)
    out = StrideIndex{Ns,permute(R, pdims),c2}(
        eachop(getmul, pdims, map(maybe_static_step, inds), s),
        permute(o, pdims)
    )
    return compose(OffsetIndex(reduce_tup(+, map(*, map(_diff, inds, o), s))), out)
end
@inline _diff(::Base.Slice, ::Any) = Zero()
@inline _diff(x::AbstractRange, o) = static_first(x) - o
@inline _diff(x::Integer, o) = x - o

@inline function compose(x::StrideIndex{1,R,C}, ::PermutedIndex{2,(2,1),(2,)}) where {R,C}
    if C === nothing
        c2 = nothing
    elseif C === 1
        c2 = 2
    else
        c2 = -1
    end
    s = getfield(strides(x), 1)
    return StrideIndex{2,(2,1),c2}((s, s), (static(1), offset1(x)))
end

@inline function compose(x::StrideIndex{N,R,C}, ::PermutedIndex{N,I1,I2}) where {N,R,C,I1,I2}
    if C === nothing || C === -1
        c2 = C
    else
        c2 = getfield(I2, C)
    end
    return StrideIndex{N,permute(R, Val(I1)),c2}(
        permute(strides(x), Val(I1)),
        permute(offsets(x), Val(I1)),
    )
end
@inline function compose(x::PermutedIndex{<:Any,I11,I12},::PermutedIndex{<:Any,I21,I22}) where {I11,I12,I21,I22}
    PermutedIndex(permute(static(I11), static(I21)), permute(static(I12), static(I22)))
end
@inline function compose(x::LinearSubIndex, y::LinearSubIndex)
    LinearSubIndex(offset1(x) + offset1(y) * stride1(x), stride1(y) * stride1(x))
end
compose(::OffsetIndex{StaticInt{0}}, y::StrideIndex) = y
compose(x::ArrayIndex, y::CartesianIndices) = ComposedIndex(x, y)

function compose(x::AbstractArray{T,N}, ::PermutedIndex{N,I1,I2}) where {T,N,I1,I2}
    PermutedDimsArray{T,N,I1,I2,typeof(x)}(x)
end
# TODO call to more direct constructors so that we don't repeat checks already performed
# when constructin SubIndex
compose(x::AbstractArray, y::SubIndex) = SubArray(x, getfield(y, :indices))
compose(x::AbstractArray, y::ComposedIndex) = compose(compose(x, outer(y)), inner(y))
compose(x::AbstractArray, y::ArrayIndex) = ComposedIndex(x, y)

## show(::IO, ::MIME, ::ArrayIndex)
function Base.show(io::IO, ::MIME"text/plain", @nospecialize(x::StrideIndex))
    print(io, "StrideIndex{$(ndims(x)), $(known(stride_rank(x))), $(known(contiguous_axis(x)))}($(strides(x)), $(offsets(x)))")
end
function Base.show(io::IO, ::MIME"text/plain", @nospecialize(x::SubIndex))
    print(io, "SubIndex{$(ndims(x))}($(x.indices))")
end
function Base.show(io::IO, ::MIME"text/plain", @nospecialize(x::LinearSubIndex))
    print(io, "LinearSubIndex(offset=$(offset1(x)),stride=$(stride1(x)))")
end
function Base.show(io::IO, m::MIME"text/plain", @nospecialize(x::ComposedIndex))
    show(io, m, outer(x))
    print(io, " ∘ ")
    show(io, m, inner(x))
end


