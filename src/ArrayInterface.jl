module ArrayInterface

using Requires
using LinearAlgebra
using SparseArrays

function ismutable end

"""
    ismutable(x::DataType)

Query whether a type is mutable or not, see
https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/19.
"""
Base.@pure ismutable(x::DataType) = x.mutable
ismutable(x) = ismutable(typeof(x))

ismutable(::Type{<:Array}) = true
ismutable(::Type{<:Number}) = false

"""
    can_setindex(x::DataType)

Query whether a type can use `setindex!`
"""
can_setindex(x) = true
can_setindex(x::AbstractArray) = can_setindex(typeof(x))

"""
    fast_scalar_indexing(x)

Query whether an array type has fast scalar indexing
"""
fast_scalar_indexing(x) = true
fast_scalar_indexing(x::AbstractArray) = fast_scalar_indexing(typeof(x))

"""
    isstructured(x::DataType)

Query whether a type is a representation of a structured matrix
"""
isstructured(x) = false
isstructured(x::AbstractArray) = isstructured(typeof(x))
isstructured(::Symmetric) = true
isstructured(::Hermitian) = true
isstructured(::UpperTriangular) = true
isstructured(::LowerTriangular) = true
isstructured(::Tridiagonal) = true
isstructured(::SymTridiagonal) = true
isstructured(::Bidiagonal) = true
isstructured(::Diagonal) = true

"""
    has_sparsestruct(x::AbstractArray)

determine whether `findstructralnz` accepts the parameter `x`
"""
has_sparsestruct(x) = false
has_sparsestruct(x::AbstractArray) = has_sparsestruct(typeof(x))
has_sparsestruct(x::Type{<:AbstractArray}) = false
has_sparsestruct(x::Type{<:SparseMatrixCSC}) = true
has_sparsestruct(x::Type{<:Diagonal}) = true
has_sparsestruct(x::Type{<:Bidiagonal}) = true
has_sparsestruct(x::Type{<:Tridiagonal}) = true
has_sparsestruct(x::Type{<:SymTridiagonal}) = true

"""
    findstructralnz(x::AbstractArray)

Return: (I,J) #indexable objects
Find sparsity pattern of special matrices, the same as the first two elements of findnz(::SparseMatrixCSC)
"""
function findstructralnz(x::Diagonal)
  n=size(x,1)
  (1:n,1:n)
end

abstract type MatrixIndex end

struct BidiagonalIndex <: MatrixIndex
  count::Int
  isup::Bool
end

struct TridiagonalIndex <: MatrixIndex
  count::Int
  nsize::Int
  isrow::Bool
end

Base.firstindex(ind::MatrixIndex)=1
Base.lastindex(ind::MatrixIndex)=ind.count
Base.length(ind::MatrixIndex)=ind.count
function Base.getindex(ind::BidiagonalIndex,i::Int)
  1 <= i <= ind.count || throw(BoundsError(ind, i))
  if ind.isup
    ii=i+1
  else
    ii=i+1+1
  end
  convert(Int,floor(ii/2))
end

function Base.getindex(ind::TridiagonalIndex,i::Int)
  1 <= i <= ind.count || throw(BoundsError(ind, i))
  offsetu= ind.isrow ? 0 : 1
  offsetl= ind.isrow ? 1 : 0
  if 1 <= i <= ind.nsize
    return i
  elseif ind.nsize < i <= ind.nsize+ind.nsize-1
    return i-ind.nsize+offsetu
  else
    return i-(ind.nsize+ind.nsize-1)+offsetl
  end
end

function findstructralnz(x::Bidiagonal)
  n=size(x,1)
  isup= x.uplo=='U' ? true : false
  rowind=BidiagonalIndex(n+n-1,isup)
  colind=BidiagonalIndex(n+n-1,!isup)
  (rowind,colind)
end

function findstructralnz(x::Union{Tridiagonal,SymTridiagonal})
  n=size(x,1)
  rowind=TridiagonalIndex(n+n-1+n-1,n,true)
  colind=TridiagonalIndex(n+n-1+n-1,n,false)
  (rowind,colind)
end

function findstructralnz(x::SparseMatrixCSC)
  rowind,colind,_=findnz(x)
  (rowind,colind)
end

abstract type ColoringAlgorithm end

"""
    fast_matrix_colors(A)

    Query whether a matrix has a fast algorithm for getting the structural
    colors of the matrix.
"""
fast_matrix_colors(A) = false
fast_matrix_colors(A::AbstractArray) = fast_matrix_colors(typeof(A))
fast_matrix_colors(A::Type{<:Union{Diagonal,Bidiagonal,Tridiagonal,SymTridiagonal}}) = true

"""
    matrix_colors(A::Union{Array,UpperTriangular,LowerTriangular})

    The color vector for dense matrix and triangular matrix is simply
    `[1,2,3,...,size(A,2)]`
"""
function matrix_colors(A::Union{Array,UpperTriangular,LowerTriangular})
    eachindex(1:size(A,2)) # Vector size matches number of rows
end

function _cycle(repetend,len)
    repeat(repetend,div(len,length(repetend))+1)[1:len]
end

function matrix_colors(A::Diagonal)
    fill(1,size(A,2))
end

function matrix_colors(A::Bidiagonal)
    _cycle(1:2,size(A,2))
end

function matrix_colors(A::Union{Tridiagonal,SymTridiagonal})
    _cycle(1:3,size(A,2))
end

function __init__()

  @require StaticArrays="90137ffa-7385-5640-81b9-e52037218182" begin
    ismutable(::Type{<:StaticArrays.StaticArray}) = false
    can_setindex(::Type{<:StaticArrays.StaticArray}) = false
    ismutable(::Type{<:StaticArrays.MArray}) = true
  end

  @require LabelledArrays="2ee39098-c373-598a-b85f-a56591580800" begin
    ismutable(::Type{<:LabelledArrays.LArray{T,N,Syms}}) where {T,N,Syms} = ismutable(T)
    can_setindex(::Type{<:LabelledArrays.SLArray}) = false
  end

  @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
    ismutable(::Type{<:Tracker.TrackedArray}) = false
    can_setindex(::Type{<:Tracker.TrackedArray}) = false
    fast_scalar_indexing(::Type{<:Tracker.TrackedArray}) = false
  end

  @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    fast_scalar_indexing(::Type{<:CuArrays.CuArray}) = false
  end

  @require BandedMatrices="aae01518-5342-5314-be14-df237901396f" begin
    is_structured(::Type{<:BandedMatrices.BandedMatrix}) = true
    fast_matrix_colors(::Type{<:BandedMatrices.BandedMatrix}) = true
    
    function matrix_colors(A::BandedMatrices.BandedMatrix)
        u,l=bandwidths(A)
        width=u+l+1
        _cycle(1:width,size(A,2))
    end

  end

  @require BlockBandedMatrices="aae01518-5342-5314-be14-df237901396f" begin
    is_structured(::Type{<:BandedMatrices.BlockBandedMatrix}) = true
    is_structured(::Type{<:BandedMatrices.BandedBlockBandedMatrix}) = true
    fast_matrix_colors(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
    fast_matrix_colors(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true

    function matrix_colors(A::BlockBandedMatrices.BlockBandedMatrix)
        l,u=blockbandwidths(A)
        blockwidth=l+u+1
        nblock=nblocks(A,2)
        cols=[blocksize(A,(1,i))[2] for i in 1:nblock]
        blockcolors=_cycle(1:blockwidth,nblock)
        #the reserved number of colors of a block is the maximum length of columns of blocks with the same block color
        ncolors=[maximum(cols[i:blockwidth:nblock]) for i in 1:blockwidth]
        endinds=cumsum(ncolors)
        startinds=[endinds[i]-ncolors[i]+1 for i in 1:blockwidth]
        colors=[(startinds[blockcolors[i]]:endinds[blockcolors[i]])[1:cols[i]] for i in 1:nblock]
        vcat(colors...)
    end

    function matrix_colors(A::BlockBandedMatrices.BandedBlockBandedMatrix)
        l,u=blockbandwidths(A)
        lambda,mu=subblockbandwidths(A)
        blockwidth=l+u+1
        subblockwidth=lambda+mu+1
        nblock=nblocks(A,2)
        cols=[blocksize(A,(1,i))[2] for i in 1:nblock]
        blockcolors=_cycle(1:blockwidth,nblock)
        #the reserved number of colors of a block is the min of subblockwidth and the largest length of columns of blocks with the same block color
        ncolors=[min(subblockwidth,maximum(cols[i:blockwidth:nblock])) for i in 1:min(blockwidth,nblock)]
        endinds=cumsum(ncolors)
        startinds=[endinds[i]-ncolors[i]+1 for i in 1:min(blockwidth,nblock)]
        colors=[_cycle(startinds[blockcolors[i]]:endinds[blockcolors[i]],cols[i]) for i in 1:nblock]
        vcat(colors...)
    end

  end
end

end
