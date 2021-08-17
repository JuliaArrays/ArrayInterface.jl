
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
  Equal

Equality indicator. Left generic to indicate possible other uses where it's easy to interpret.
In case of `(s::StrideIndex).offset1 === Equal()`, this means that `offset1` is
```julia
s.offset[1]
```

Use of `StaticInt`s is preferred when possible for convenience. The purpose here is simply
compressing runtime size information for when `StrideIndex` is passed as a non-inlined argument.
"""
struct Equal end
"""
    StrideIndex(x)

Subtype of `ArrayIndex` that transforms and index using stride layout information
derived from `x`.
"""
struct StrideIndex{N,R,C,S,O,O1} <: ArrayIndex{N}
    strides::S
    offsets::O
    offset1::O1

    function StrideIndex{N,R,C}(s::S, o::O, o1::O1) where {N,R,C,S,O,O1}
        return new{N,R::NTuple{N,Int},C::Int,S,O,O1}(s, o, o1)
    end
    function StrideIndex{N,R,C}(a::A) where {N,R,C,A}
        return StrideIndex{N,R,C}(strides(a),offsets(a),offset1(a))
    end
    function StrideIndex(a::A) where {A}
        return StrideIndex{ndims(A),known(stride_rank(A)), known(contiguous_axis(A))}(a)
    end
end

Base.firstindex(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = 1
Base.lastindex(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = i.count
Base.length(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = i.count
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

@inline function Base.getindex(x::StrideIndex{N}, i::AbstractCartesianIndex{N}) where {N}
    return _strides2int(offsets(x), strides(x), Tuple(i)) + offset1(x)
end
@generated function _strides2int(o::O, s::S, i::I) where {O,S,I}
    N = known_length(I)
    out = :()
    for i in 1:N
        tmp = :(((getfield(i, $i) - getfield(o, $i)) * getfield(s, $i)))
        out = ifelse(i === 1, tmp, :($out + $tmp))
    end
    return Expr(:block, Expr(:meta, :inline), out)
end
