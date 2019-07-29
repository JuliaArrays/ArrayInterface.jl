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

"""
    isstructured(x::DataType)

Query whether a type is a representation of a structured matrix
"""
isstructured(x) = false
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
has_sparsestruct(x)=false
has_sparsestruct(x::AbstractArray)=false
has_sparsestruct(x::SparseMatrixCSC)=true
has_sparsestruct(x::Diagonal)=true
has_sparsestruct(x::Bidiagonal)=true
has_sparsestruct(x::Tridiagonal)=true
has_sparsestruct(x::SymTridiagonal)=true

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
  end

  @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
    ismutable(::Type{<:Flux.Tracker.TrackedArray}) = false
    can_setindex(::Type{<:Flux.Tracker.TrackedArray}) = false
  end

  @require BandedMatrices="aae01518-5342-5314-be14-df237901396f" begin
    is_structured(::BandedMatrices.BandedMatrix) = true

    function matrix_colors(A::BandedMatrix)
        u,l=bandwidths(A)
        width=u+l+1
        _cycle(1:width,size(A,2))
    end
    
  end

  @require BlockBandedMatrices="aae01518-5342-5314-be14-df237901396f" begin
    is_structured(::BandedMatrices.BlockBandedMatrix) = true
    is_structured(::BandedMatrices.BandedBlockBandedMatrix) = true

    function matrix_colors(A::BlockBandedMatrix)
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

    function matrix_colors(A::BandedBlockBandedMatrix)
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
