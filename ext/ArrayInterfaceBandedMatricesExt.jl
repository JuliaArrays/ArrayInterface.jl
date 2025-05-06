module ArrayInterfaceBandedMatricesExt

using ArrayInterface
using ArrayInterface: BandedMatrixIndex
using BandedMatrices
using LinearAlgebra

const TransOrAdjBandedMatrix = Union{
    Adjoint{T, <:BandedMatrix{T}},
    Transpose{T, <:BandedMatrix{T}},
} where {T}

const AllBandedMatrix = Union{
    BandedMatrix{T},
    TransOrAdjBandedMatrix{T},
} where {T}

Base.firstindex(i::BandedMatrixIndex) = 1
Base.lastindex(i::BandedMatrixIndex) = i.count
Base.length(i::BandedMatrixIndex) = lastindex(i)
Base.@propagate_inbounds function Base.getindex(ind::BandedMatrixIndex, i::Int)
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

function ArrayInterface.BandedMatrixIndex(rowsize, colsize, lowerbandwidth, upperbandwidth, isrow)
    upperbandwidth > -lowerbandwidth || throw(ErrorException("Invalid Bandwidths"))
    bandinds = upperbandwidth:-1:(-lowerbandwidth)
    bandsizes = [_bandsize(band, rowsize, colsize) for band in bandinds]
    BandedMatrixIndex(sum(bandsizes), rowsize, colsize, bandinds, bandsizes, isrow)
end

function ArrayInterface.findstructralnz(x::AllBandedMatrix)
    l, u = BandedMatrices.bandwidths(x)
    rowsize, colsize = Base.size(x)
    rowind = BandedMatrixIndex(rowsize, colsize, l, u, true)
    colind = BandedMatrixIndex(rowsize, colsize, l, u, false)
    return (rowind, colind)
end

ArrayInterface.has_sparsestruct(::Type{<:AllBandedMatrix}) = true
ArrayInterface.isstructured(::Type{<:AllBandedMatrix}) = true
ArrayInterface.fast_matrix_colors(::Type{<:AllBandedMatrix}) = true

function ArrayInterface.matrix_colors(A::AllBandedMatrix)
    l, u = BandedMatrices.bandwidths(A)
    width = u + l + 1
    return ArrayInterface._cycle(1:width, Base.size(A, 2))
end

end # module
