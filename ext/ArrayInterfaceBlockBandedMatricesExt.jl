module ArrayInterfaceBlockBandedMatricesExt



if isdefined(Base, :get_extension)
    using ArrayInterface
    using ArrayInterface: BandedMatrixIndex
    using BlockBandedMatrices
    using BlockBandedMatrices.BlockArrays
else
    using ..ArrayInterface
    using ..ArrayInterface: BandedMatrixIndex
    using ..BlockBandedMatrices
    using ..BlockBandedMatrices.BlockArrays
end

struct BlockBandedMatrixIndex <: ArrayInterface.MatrixIndex
    count::Int
    refinds::Array{Int,1}
    refcoords::Array{Int,1}# storing col or row inds at ref points
    isrow::Bool
end
Base.firstindex(i::BlockBandedMatrixIndex) = 1
Base.lastindex(i::BlockBandedMatrixIndex) = i.count
Base.length(i::BlockBandedMatrixIndex) = lastindex(i)
function BlockBandedMatrixIndex(nrowblock, ncolblock, rowsizes, colsizes, l, u)
    blockrowind = BandedMatrixIndex(nrowblock, ncolblock, l, u, true)
    blockcolind = BandedMatrixIndex(nrowblock, ncolblock, l, u, false)
    sortedinds = sort(
        [(blockrowind[i], blockcolind[i]) for i = 1:length(blockrowind)],
        by=x -> x[1],
    )
    sort!(sortedinds, by=x -> x[2], alg=InsertionSort)# stable sort keeps the second index in order
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
Base.@propagate_inbounds function Base.getindex(ind::BlockBandedMatrixIndex, i::Int)
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

function ArrayInterface.findstructralnz(x::BlockBandedMatrices.BlockBandedMatrix)
    l, u = BlockBandedMatrices.blockbandwidths(x)
    nrowblock = BlockBandedMatrices.blocksize(x, 1)
    ncolblock = BlockBandedMatrices.blocksize(x, 2)
    rowsizes = BlockArrays.blocklengths(axes(x, 1))
    colsizes = BlockArrays.blocklengths(axes(x, 2))
    return BlockBandedMatrixIndex(
        nrowblock,
        ncolblock,
        rowsizes,
        colsizes,
        l,
        u,
    )
end
struct BandedBlockBandedMatrixIndex <: ArrayInterface.MatrixIndex
    count::Int
    refinds::Array{Int,1}
    refcoords::Array{Int,1}# storing col or row inds at ref points
    reflocalinds::Array{BandedMatrixIndex,1}
    isrow::Bool
end
Base.firstindex(i::BandedBlockBandedMatrixIndex) = 1
Base.lastindex(i::BandedBlockBandedMatrixIndex) = i.count
Base.length(i::BandedBlockBandedMatrixIndex) = lastindex(i)
Base.@propagate_inbounds function Base.getindex(ind::BandedBlockBandedMatrixIndex, i::Int)
    @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
    p = 1
    while i - ind.refinds[p] >= 0
        p += 1
    end
    p -= 1
    _i = i - ind.refinds[p] + 1
    ind.reflocalinds[p][_i] + ind.refcoords[p] - 1
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
        by=x -> x[1],
    )
    sort!(sortedinds, by=x -> x[2], alg=InsertionSort)# stable sort keeps the second index in order
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

function ArrayInterface.findstructralnz(x::BlockBandedMatrices.BandedBlockBandedMatrix)
    l, u = BlockBandedMatrices.blockbandwidths(x)
    lambda, mu = BlockBandedMatrices.subblockbandwidths(x)
    nrowblock = BlockBandedMatrices.blocksize(x, 1)
    ncolblock = BlockBandedMatrices.blocksize(x, 2)
    rowsizes = BlockArrays.blocklengths(axes(x, 1))
    colsizes = BlockArrays.blocklengths(axes(x, 2))
    return BandedBlockBandedMatrixIndex(
        nrowblock,
        ncolblock,
        rowsizes,
        colsizes,
        l,
        u,
        lambda,
        mu,
    )
end

ArrayInterface.has_sparsestruct(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
ArrayInterface.has_sparsestruct(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true
ArrayInterface.isstructured(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
ArrayInterface.isstructured(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true
ArrayInterface.fast_matrix_colors(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
ArrayInterface.fast_matrix_colors(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true

function ArrayInterface.matrix_colors(A::BlockBandedMatrices.BlockBandedMatrix)
    l, u = BlockBandedMatrices.blockbandwidths(A)
    blockwidth = l + u + 1
    nblock = BlockBandedMatrices.blocksize(A, 2)
    cols = BlockArrays.blocklengths(axes(A, 2))
    blockcolors = ArrayInterface._cycle(1:blockwidth, nblock)
    # the reserved number of colors of a block is the maximum length of columns of blocks with the same block color
    ncolors = [maximum(cols[i:blockwidth:nblock]) for i = 1:blockwidth]
    endinds = cumsum(ncolors)
    startinds = [endinds[i] - ncolors[i] + 1 for i = 1:blockwidth]
    colors = [
        (startinds[blockcolors[i]]:endinds[blockcolors[i]])[1:cols[i]]
        for i = 1:nblock
    ]
    return reduce(vcat, colors)
end

function ArrayInterface.matrix_colors(A::BlockBandedMatrices.BandedBlockBandedMatrix)
    l, u = BlockBandedMatrices.blockbandwidths(A)
    lambda, mu = BlockBandedMatrices.subblockbandwidths(A)
    blockwidth = l + u + 1
    subblockwidth = lambda + mu + 1
    nblock = BlockBandedMatrices.blocksize(A, 2)
    cols = BlockArrays.blocklengths(axes(A, 2))
    blockcolors = ArrayInterface._cycle(1:blockwidth, nblock)
    # the reserved number of colors of a block is the min of subblockwidth and the largest length of columns of blocks with the same block color
    ncolors = [
        min(subblockwidth, maximum(cols[i:blockwidth:nblock]))
        for i = 1:min(blockwidth, nblock)
    ]
    endinds = cumsum(ncolors)
    startinds = [endinds[i] - ncolors[i] + 1 for i = 1:min(blockwidth, nblock)]
    colors = [
        ArrayInterface._cycle(startinds[blockcolors[i]]:endinds[blockcolors[i]], cols[i])
        for i = 1:nblock
    ]
    return reduce(vcat, colors)
end

end # module
