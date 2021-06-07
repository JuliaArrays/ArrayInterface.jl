
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

Base.getindex(x::ArrayIndex{N}, i::Vararg{CanonicalInt,N}) where {N} = x[NDIndex(i)]

struct LinkedIndex{N,I<:Tuple} <: ArrayIndex{N}
    index::I

    LinkedIndex(x::Tuple{I,Vararg{Any}}) where {I} = new{ndims(I),typeof(x)}(x)
end

link_index(f::Tuple) = Base.Fix2(link_index, f)
link_index(x, f::Tuple) = LinkedIndex(_link_index(x, f))
_link_index(x, f::Tuple{Any,Vararg{Any}}) = (first(f)(x), _link_index(x, tail(f))...)
_link_index(x, f::Tuple{Any}) = (first(f)(x),)
_link_index(x, f::Tuple{}) = ()

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
    refcoords::Array{Int,1}  # storing col or row inds at ref points
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

struct StrideIndex{N,R,C,S,O,O1} <: ArrayIndex{N}
    strides::S
    offsets::O
    offset1::O1

    function StrideIndex{N,R,C,S,O,O1}(s::S, o::O, o1::O1) where {N,R,C,S<:Tuple{Vararg{<:CanonicalInt,N}},O<:Tuple{Vararg{<:CanonicalInt,N}},O1}
        new{N,R::NTuple{N,Int},C::Int,S,O,O1}(s, o, o1)
    end
    function StrideIndex{N,R,C,S,O,O1}(x) where {N,R,C,S<:Tuple{Vararg{<:CanonicalInt,N}},O<:Tuple{Vararg{<:CanonicalInt,N}},O1}
        new{N,R::NTuple{N,Int},C::Int,S,O,O1}(strides(x), offsets(x), offset1(x))
    end
    function StrideIndex{N,R,C}(s::S, o::O, o1::O1) where {N,R,C,S<:Tuple{Vararg{<:CanonicalInt,N}},O<:Tuple{Vararg{<:CanonicalInt,N}},O1}
        new{N,R::NTuple{N,Int},C::Int,S,O,O1}(s, o, o1)
    end
    function StrideIndex{N,R,C}(a::A) where {N,R,C,A}
        StrideIndex{N,R,C}(strides(a),offsets(a),offset1(a))
    end
    function StrideIndex(a::A) where {A}
        StrideIndex{ndims(A),known(stride_rank(A)), known(contiguous_axis(A))}(a)
    end
end

"""
    LinearIndex

Subtypes of `ArrayIndex` that transforms an index by by a common offset (`o1`).
"""
struct LinearIndex{O<:CanonicalInt} <: VectorIndex
    offset1::O

    LinearIndex{O}(x::O) where {O} = new{O}(x)
    LinearIndex{O}(x::DenseArray) where {O} = LinearIndex{O}(static(0))
end

"""
    PermutedIndex

Subtypes of `ArrayIndex` that is responsible for permuting each index prior to accessing
parent indices.
"""
struct PermutedIndex{N,perm} <: ArrayIndex{N}

    PermutedIndex{N,perm}() where {N,perm} = new{N,perm}()
    PermutedIndex{N,perm}(x::AbstractArray) where {N,perm} = new{N,perm}()
end

"""
    SubIndex

Subtypes of `ArrayIndex` that provides a multidimensional view of another `ArrayIndex`.
"""
struct SubIndex{N,I} <: ArrayIndex{N}
    indices::I

    SubIndex{N,I}(x::SubArray) where {N,I} = new{N,I}(x.indices)
end

"""
    LinearSubIndex{S,I}

Subtypes of `ArrayIndex` that provides a multidimensional view of another `ArrayIndex`.
"""
struct LinearSubIndex{S,I} <: VectorIndex
    stride1::S
    offset1::I

    function LinearSubIndex{StaticInt{S},StaticInt{O}}() where {S,O}
        new{StaticInt{S},StaticInt{O}}(StaticInt{S}(),StaticInt{O}())
    end
    LinearSubIndex{Int,Int}(x::SubArray) = new{S,I}(x.stride1, x.offset1)
end

"""
    LinearNDIndex(offset1, offsets, size)

A linear representation of a multidimensional index. Unlike `CartesianIndices`, indexing
`LinearNDIndex` by an `StaticInt` can produce a static type (e.g., `NDIndex`).
"""
struct LinearNDIndex{N,O1,O<:Tuple{Vararg{CanonicalInt,N}},S<:Tuple{Vararg{CanonicalInt,N}}} <: VectorIndex
    offset1::O1
    offsets::O
    size::S

    function LinearNDIndex{N,O1,O,S}(x::AbstractArray) where {N,O1,O,S}
        new{N,O1,O,S}(offset1(x), offsets(x), size(x))
    end
end

struct NDLinearIndex{N,O1,O<:Tuple{Vararg{CanonicalInt,N}},S<:Tuple{Vararg{CanonicalInt,N}}} <: ArrayIndex{N}
    offset1::O1
    offsets::O
    size::S

    function NDLinearIndex{N,O1,O,S}(x::AbstractArray) where {N,O1,O,S}
        new{N,O1,O,S}(offset1(x), offsets(x), size(x))
    end
end

struct TransposedVectorIndex <: ArrayIndex{2}
    TransposedVectorIndex() = new()
end

is_static(::Type{<:ArrayIndex}) = static(false)
is_static(::Type{TransposedVectorIndex}) = static(true)
is_static(::Type{LinearNDIndex{N,StaticInt,Tuple{Vararg{StaticInt,N}},Tuple{Vararg{StaticInt,N}}}}) where {N} = static(true)
is_static(::Type{NDLinearIndex{N,StaticInt,Tuple{Vararg{StaticInt,N}},Tuple{Vararg{StaticInt,N}}}}) where {N} = static(true)
is_static(::Type{StrideIndex{N,R,C,Tuple{Vararg{StaticInt,N}},Tuple{Vararg{StaticInt,N}},StaticInt}}) where {N,R,C} = static(true)
is_static(::Type{LinearIndex{StaticInt}}) = static(true)
is_static(::Type{LinearSubIndex{StaticInt,StaticInt}}) = static(true)

Base.firstindex(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = 1
Base.lastindex(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = i.count
Base.length(i::Union{TridiagonalIndex,BandedBlockBandedMatrixIndex,BandedMatrixIndex,BidiagonalIndex,BlockBandedMatrixIndex}) = i.count
function Base.getindex(ind::BidiagonalIndex, i::Int)
    1 <= i <= ind.count || throw(BoundsError(ind, i))
    if ind.isup
        ii = i + 1
    else
        ii = i + 1 + 1
    end
    convert(Int, floor(ii / 2))
end

function Base.getindex(ind::TridiagonalIndex, i::Int)
    1 <= i <= ind.count || throw(BoundsError(ind, i))
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

function Base.getindex(ind::BandedMatrixIndex, i::Int)
    1 <= i <= ind.count || throw(BoundsError(ind, i))
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

function Base.getindex(ind::BlockBandedMatrixIndex, i::Int)
    1 <= i <= ind.count || throw(BoundsError(ind, i))
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

function Base.getindex(ind::BandedBlockBandedMatrixIndex, i::Int)
    1 <= i <= ind.count || throw(BoundsError(ind, i))
    p = 1
    while i - ind.refinds[p] >= 0
        p += 1
    end
    p -= 1
    _i = i - ind.refinds[p] + 1
    ind.reflocalinds[p][_i] + ind.refcoords[p] - 1
end

@inline function Base.getindex(x::StrideIndex{N}, i::AbstractCartesianIndex{N}) where {N}
    return _strides2int(x.offsets, x.strides, Tuple(i)) + x.offset1
end
@inline function _strides2int(o::Tuple, s::Tuple, i::Tuple)
    return ((first(i) - first(o)) * first(s)) + _strides2int(tail(o), tail(s), tail(i))
end
_strides2int(::Tuple{}, ::Tuple{}, ::Tuple{}) = static(0)

@inline function Base.getindex(x::PermutedIndex{N,perm}, i::AbstractCartesianIndex{N}) where {N,perm}
    return NDIndex(permute(i, Val(perm)))
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
    while (inds_i <= NI) && (subinds_i <= NS)
        subinds_type = S.parameters[subinds_i]
        if subinds_type <: Integer
            push!(out.args, :(getfield(subinds, $subinds_i)))
            subinds_i += 1
        elseif subinds_type <: Slice
            push!(out.args, :(getfield(inds, $inds_i)))
            inds_i += 1
            subinds_i += 1
        else
            T_i = eltype(subinds_type)
            if T_i <: AbstractCartesianIndex
                push!(out.args, :(Tuple(@inbounds(getfield(subinds, $subinds_i)[getfield(subinds, $inds_i)]))...))
                inds_i += 1
                subinds_i += 1
            else
                push!(out.args, :(Tuple(@inbounds(getfield(subinds, $subinds_i)[getfield(subinds, $inds_i)]))))
                inds_i += 1
                subinds_i += 1
            end
        end
    end
    return Expr(:block, Expr(:meta, :inline), :($out))
end

@inline function Base.getindex(x::NDLinearIndex{N}, i::AbstractCartesianIndex{N}) where {N}
    inds = Tuple(arg)
    o = offsets(x)
    s = size(x)
    return first(inds) + (offset1(x) - first(o)) + _subs2lin(first(s), tail(s), tail(o), tail(inds))
end
@inline function _subs2lin(str, s::Tuple{Any,Vararg}, o::Tuple{Any,Vararg}, i::Tuple{Any,Vararg})
    return ((first(i) - first(o)) * str) + _subs2lin(str * first(s), tail(s), tail(o), tail(i))
end
_subs2lin(str, s::Tuple{Any}, o::Tuple{Any}, i::Tuple{Any}) = (first(i) - first(o)) * str
# trailing inbounds can only be 1 or 1:1
_subs2lin(str, ::Tuple{}, ::Tuple{}, ::Tuple{Any}) = static(0)

@inline function Base.getindex(x::LinearNDIndex, i::CanonicalInt)
    return NDIndex(_lin2subs(x.offsets, x.size, i - x.offset1))
end
@inline function _lin2subs(o::Tuple{Any,Vararg{Any}}, s::Tuple{Any,Vararg{Any}}, i::CanonicalInt)
    len = first(s)
    inext = div(i, len)
    return (i - len * inext + first(o), _lin2subs(tail(o), tail(s), inext)...)
end
_lin2subs(o::Tuple{Any}, s::Tuple{Any}, i::CanonicalInt) = i + first(o)

Base.getindex(x::LinearIndex, i::CanonicalInt) = x.offset1 + i

Base.getindex(x::TransposedVectorIndex, i::AbstractCartesianIndex{2}) = last(Tuple(i))

Base.getindex(x::LinearSubIndex, i::CanonicalInt) =  (x.strid1 * i) + x.offset1

function Base.getindex(x::LinkedIndex{N}, i::AbstractCartesianIndex{N}) where {N}
    return _get_linked_index(x.index, i)
end
Base.getindex(x::LinkedIndex{1}, i::CanonicalInt) = _get_linked_index(x.index, i)

@generated function _get_linked_index(index::I, i) where {N,I<:Tuple{Vararg{Any,N}}}
    out = :(@inbounds(getfield(index, 1)[i]))
    if N > 1
        for n in 2:N
            out = :(@inbounds(getfield(index, $n)[$out]))
        end
    end
    Expr(:block, Expr(:meta, :inline), out)
end
