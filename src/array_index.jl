
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

Base.firstindex(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = 1
Base.lastindex(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = i.count
Base.length(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = i.count

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

function Base.getindex(x::PermutedIndex{2,(2,1),(2,)}, i::AbstractCartesianIndex{2})
    getfield(Tuple(i), 2)
end
@inline function Base.getindex(x::PermutedIndex{N,I1,I2}, i::AbstractCartesianIndex{N}) where {N,I1,I2}
    return NDIndex(permute(Tuple(i), Val(I2)))
end

"""
    SubIndex(indices)

Subtype of `ArrayIndex` that provides a multidimensional view of another `ArrayIndex`.
"""
struct SubIndex{N,I} <: ArrayIndex{N}
    indices::I

    SubIndex{N}(inds::Tuple) where {N} = new{N,typeof(inds)}(inds)
end

@inline function Base.getindex(x::SubIndex{N}, i::AbstractCartesianIndex{N}) where {N}
    return NDIndex(_reindex(x.indices, Tuple(i)))
end
@generated function _reindex(subinds::S, inds::I) where {S,I}
    inds_i = 1
    subinds_i = 1
    NS = known_length(S)
    NI = known_length(I)
    out = Expr(:tuple)
    while inds_i <= NI
        subinds_type = S.parameters[subinds_i]
        if subinds_type <: Integer
            push!(out.args, :(getfield(subinds, $subinds_i)))
            subinds_i += 1
        elseif eltype(subinds_type) <: AbstractCartesianIndex
            push!(out.args, :(Tuple(@inbounds(getfield(subinds, $subinds_i)[getfield(inds, $inds_i)]))...))
            inds_i += 1
            subinds_i += 1
        else
            push!(out.args, :(@inbounds(getfield(subinds, $subinds_i)[getfield(inds, $inds_i)])))
            inds_i += 1
            subinds_i += 1
        end
    end
    if subinds_i <= NS
        for i in subinds_i:NS
            push!(out.args, :(getfield(subinds, $subinds_i)))
        end
    end
    return Expr(:block, Expr(:meta, :inline), :($out))
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

const OffsetIndex{O} = LinearSubIndex{O,StaticInt{1}}
OffsetIndex(offset::CanonicalInt) = LinearSubIndex(offset, static(1))

@inline function Base.getindex(x::LinearSubIndex, i::CanonicalInt)
    getfield(x, :offset) + getfield(x, :stride) * i
end

"""
    ComposedIndex(i1, i2)

A subtype of `ArrayIndex` that lazily combines index `i1` and `i2`. Indexing a
`ComposedIndex` whith `i` is equivalent to `i2[i1[i]]`.
"""
struct ComposedIndex{N,I1,I2} <: ArrayIndex{N}
    i1::I1
    i2::I2

    ComposedIndex(i1::I1, i2::I2) where {I1,I2} = new{ndims(I1),I1,I2}(i1, i2)
end
# we should be able to assume that if `i1` was indexed without error than it's inbounds
@propagate_inbounds function Base.getindex(x::ComposedIndex)
    ii = getfield(x, :i1)[]
    @inbounds(getfield(x, :i2)[ii])
end
@propagate_inbounds function Base.getindex(x::ComposedIndex, i::CanonicalInt)
    ii = getfield(x, :i1)[i]
    @inbounds(getfield(x, :i2)[ii])
end
@propagate_inbounds function Base.getindex(x::ComposedIndex, i::AbstractCartesianIndex)
    ii = getfield(x, :i1)[i]
    @inbounds(getfield(x, :i2)[ii])
end

Base.getindex(x::ArrayIndex, i::ArrayIndex) = ComposedIndex(i, x)
@inline function Base.getindex(x::ComposedIndex, i::ArrayIndex)
    ComposedIndex(getfield(x, :i1)[i], getfield(x, :i2))
end
@inline function Base.getindex(x::ArrayIndex, i::ComposedIndex)
    ComposedIndex(getfield(i, :i1), x[getfield(i, :i2)])
end
@inline function Base.getindex(x::ComposedIndex, i::ComposedIndex)
    ComposedIndex(getfield(i, :i1), ComposedIndex(getfield(x, :i1)[getfield(i, :i2)], getfield(x, :i2)))
end

@propagate_inbounds Base.getindex(x::ArrayIndex, i::CanonicalInt, ii::CanonicalInt...) = x[NDIndex(i, ii...)]
@propagate_inbounds function Base.getindex(ind::BidiagonalIndex, i::Int)
    @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
    if ind.isup
        ii = i + 1
    else
        ii = i + 1 + 1
    end
    convert(Int, floor(ii / 2))
end

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

@inline function Base.getindex(x::StrideIndex{N}, i::AbstractCartesianIndex) where {N}
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

@inline function Base.getindex(x::StrideIndex, i::SubIndex{N,I}) where {N,I}
    _composed_sub_strides(stride_preserving_index(I), x, i)
end
_composed_sub_strides(::False, x::StrideIndex, i::SubIndex) = ComposedIndex(i, x)
@inline function _composed_sub_strides(::True, x::StrideIndex{N,R,C}, i::SubIndex{Ns,I}) where {N,R,C,Ns,I<:Tuple{Vararg{Any,N}}}
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
    return OffsetIndex(reduce_tup(+, map(*, map(_diff, inds, o), s)))[out]
end
@inline _diff(::Base.Slice, ::Any) = Zero()
@inline _diff(x::AbstractRange, o) = static_first(x) - o
@inline _diff(x::Integer, o) = x - o

@inline function Base.getindex(x::StrideIndex{1,R,C}, ::PermutedIndex{2,(2,1),(2,)}) where {R,C}
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
@inline function Base.getindex(x::StrideIndex{N,R,C}, ::PermutedIndex{N,perm,iperm}) where {N,R,C,perm,iperm}
    if C === nothing || C === -1
        c2 = C
    else
        c2 = getfield(iperm, C)
    end
    return StrideIndex{N,permute(R, Val(perm)),c2}(
        permute(strides(x), Val(perm)),
        permute(offsets(x), Val(perm)),
    )
end
@inline function Base.getindex(x::PermutedIndex, i::PermutedIndex)
    PermutedIndex(
        permute(to_parent_dims(x), to_parent_dims(i)),
        permute(from_parent_dims(x), from_parent_dims(i))
    )
end

@inline function Base.getindex(x::LinearSubIndex, i::LinearSubIndex)
    s = getfield(x, :stride)
    LinearSubIndex(
        getfield(x, :offset) + getfield(i, :offset) * s,
        getfield(i, :stride) * s
    )
end
Base.getindex(::OffsetIndex{StaticInt{0}}, i::StrideIndex) = i


## ArrayIndex constructorrs
@inline _to_cartesian(a) = CartesianIndices(ntuple(dim -> indices(a, dim), Val(ndims(a))))
@inline function _to_linear(a)
    N = ndims(a)
    StrideIndex{N,ntuple(+, Val(N)),nothing}(size_to_strides(size(a), static(1)), offsets(a))
end

## DenseArray
"""
    ArrayIndex{N}(A) -> index

Constructs a subtype of `ArrayIndex` such that an `N` dimensional indexing argument may be
converted to an appropriate state for accessing the buffer of `A`. For example:

```julia
julia> A = reshape(1:20, 4, 5);

julia> index = ArrayInterface.ArrayIndex{2}(A);

julia> ArrayInterface.buffer(A)[index[2, 2]] == A[2, 2]
true

```
"""
ArrayIndex{N}(x::DenseArray) where {N} = StrideIndex(x)
ArrayIndex{1}(x::DenseArray) = OffsetIndex(static(0))

ArrayIndex{1}(x::ReshapedArray) = OffsetIndex(static(0))
ArrayIndex{N}(x::ReshapedArray) where {N} = _to_linear(x)

ArrayIndex{1}(x::AbstractRange) = OffsetIndex(static(0))

## SubArray
ArrayIndex{N}(x::SubArray) where {N} = SubIndex{ndims(x)}(getfield(x, :indices))
function ArrayIndex{1}(x::SubArray{<:Any,N}) where {N}
    ComposedIndex(_to_cartesian(x), SubIndex{N}(getfield(x, :indices)))
end
ArrayIndex{1}(x::Base.FastContiguousSubArray) = OffsetIndex(getfield(x, :offset1))
function ArrayIndex{1}(x::Base.FastSubArray)
    LinearSubIndex(getfield(x, :offset1), getfield(x, :stride1))
end

## Permuted arrays
ArrayIndex{2}(::MatAdjTrans) = PermutedIndex{2,(2,1),(2,1)}()
ArrayIndex{2}(::VecAdjTrans) = PermutedIndex{2,(2,1),(2,)}()
ArrayIndex{1}(x::MatAdjTrans) = ComposedIndex(_to_cartesian(x), ArrayIndex{2}(x))
ArrayIndex{1}(x::VecAdjTrans) = OffsetIndex(static(0))  # just unwrap permuting struct
ArrayIndex{1}(::PermutedDimsArray{<:Any,1}) = OffsetIndex(static(0))
function ArrayIndex{N}(::PermutedDimsArray{<:Any,N,perm,iperm}) where {N,perm,iperm}
    PermutedIndex{N,perm,iperm}()
end
function ArrayIndex{1}(x::PermutedDimsArray{<:Any,N,perm,iperm}) where {N,perm,iperm}
    ComposedIndex(_to_cartesian(x), PermutedIndex{N,perm,iperm}())
end

