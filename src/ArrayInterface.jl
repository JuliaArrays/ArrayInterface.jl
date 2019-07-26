module ArrayInterface

using Requires
using LinearAlgebra
using SparseArrays

export findstructralnz,has_sparsestruct

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
    has_sparsestruct(x::AbstractArray)

determine whether `findstructralnz` accepts the parameter `x`
"""
has_sparsestruct(x::AbstractArray)=false
has_sparsestruct(x::SparseMatrixCSC)=true
has_sparsestruct(x::Diagonal)=true
has_sparsestruct(x::Bidiagonal)=true
has_sparsestruct(x::Tridiagonal)=true
has_sparsestruct(x::SymTridiagonal)=true

"""
    findstructralnz(x::AbstractArray)

Return: (I,J) #indexable objects
Find sparsity pattern of special matrices, similar to first two elements of findnz(::SparseMatrixCSC)
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

function __init__()

  @require StaticArrays="90137ffa-7385-5640-81b9-e52037218182" begin
    ismutable(::Type{<:StaticArrays.StaticArray}) = false
    ismutable(::Type{<:StaticArrays.MArray}) = true
  end

  @require LabelledArrays="2ee39098-c373-598a-b85f-a56591580800" begin
    ismutable(::Type{<:LabelledArrays.LArray{T,N,Syms}}) where {T,N,Syms} = ismutable(T)
  end
  
  @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
    ismutable(::Type{<:Flux.Tracker.TrackedArray}) = false
  end
end

end
