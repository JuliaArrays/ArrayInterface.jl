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
ismutable(x) = ismutable(typeof(x))

ismutable(::Type{<:AbstractArray}) = true
ismutable(::Type{<:Number}) = false
ismutable(::Type{<:AbstractRange}) = false
ismutable(::Type{<:Tuple}) = false

# Piracy
function Base.setindex(x::AbstractArray,v,i...)
  _x = copy(x)
  _x[i...] = v
  _x
end

function Base.setindex(x::AbstractVector,v,i::Int)
  n = length(x)
  x .* (i .!== 1:n) .+ v .* (i .== 1:n)
end

function Base.setindex(x::AbstractMatrix,v,i::Int,j::Int)
  n,m = size(x)
  x .* (i .!== 1:n) .* (j  .!== i:m)' .+ v .* (i .== 1:n) .* (j  .== i:m)'
end

"""
    can_setindex(x::DataType)

Query whether a type can use `setindex!`
"""
can_setindex(x) = true
can_setindex(x::AbstractArray) = can_setindex(typeof(x))
can_setindex(::Type{<:AbstractRange}) = false

"""
    aos_to_soa(x)

Converts an array of structs formulation to a struct of array
"""
aos_to_soa(x) = x

"""
    fast_scalar_indexing(x)

Query whether an array type has fast scalar indexing
"""
fast_scalar_indexing(x) = true
fast_scalar_indexing(x::AbstractArray) = fast_scalar_indexing(typeof(x))
fast_scalar_indexing(::Type{<:LinearAlgebra.AbstractQ}) = false
fast_scalar_indexing(::Type{<:LinearAlgebra.LQPackedQ}) = false

"""
    allowed_getindex(x,i...)

A scalar getindex which is always allowed
"""
allowed_getindex(x,i...) = x[i...]

"""
    allowed_setindex!(x,v,i...)

A scalar setindex! which is always allowed
"""
allowed_setindex!(x,v,i...) = setindex!(x,v,i...)

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

Determine whether `findstructralnz` accepts the parameter `x`
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
    issingular(A::AbstractMatrix)

Determine whether a given abstract matrix is singular.
"""
issingular(A::AbstractMatrix) = issingular(Matrix(A))
issingular(A::AbstractSparseMatrix) = !issuccess(lu(A, check=false))
issingular(A::Matrix) = !issuccess(lu(A, check=false))
issingular(A::UniformScaling) = A.λ == 0
issingular(A::Diagonal) = any(iszero,A.diag)
issingular(B::Bidiagonal) = any(iszero, A.dv)
issingular(S::SymTridiagonal) = diaganyzero(iszero, ldlt(S).data)
issingular(T::Tridiagonal) = !issuccess(lu(A, check=false))
issingular(A::Union{Hermitian,Symmetric}) = diaganyzero(bunchkaufman(A, check=false).LD)
issingular(A::Union{LowerTriangular,UpperTriangular}) = diaganyzero(A.data)
issingular(A::Union{UnitLowerTriangular,UnitUpperTriangular}) = false
issingular(A::Union{Adjoint,Transpose}) = issingular(parent(A))
diaganyzero(A) = any(iszero, view(A, diagind(A)))

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
  count::Int#count==nsize+nsize-1+nsize-1
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

function _bandsize(bandind,rowsize,colsize)
  -(rowsize-1) <= bandind <= colsize-1 || throw(ErrorException("Invalid Bandind"))
  if (bandind*(colsize-rowsize)>0) & (abs(bandind)<=abs(colsize-rowsize))
    return min(rowsize,colsize)
  elseif bandind*(colsize-rowsize)<=0
    return min(rowsize,colsize)-abs(bandind)
  else
    return min(rowsize,colsize)-abs(bandind)+abs(colsize-rowsize)
  end
end

function BandedMatrixIndex(rowsize,colsize,lowerbandwidth,upperbandwidth,isrow)
  upperbandwidth>-lowerbandwidth || throw(ErrorException("Invalid Bandwidths"))
  bandinds=upperbandwidth:-1:-lowerbandwidth
  bandsizes=[_bandsize(band,rowsize,colsize) for band in bandinds]
  BandedMatrixIndex(sum(bandsizes),rowsize,colsize,bandinds,bandsizes,isrow)
end

struct BlockBandedMatrixIndex <: MatrixIndex
  count::Int
  refinds::Array{Int,1}
  refcoords::Array{Int,1}#storing col or row inds at ref points
  isrow::Bool
end

function BlockBandedMatrixIndex(nrowblock,ncolblock,rowsizes,colsizes,l,u)
  blockrowind=BandedMatrixIndex(nrowblock,ncolblock,l,u,true)
  blockcolind=BandedMatrixIndex(nrowblock,ncolblock,l,u,false)
  sortedinds=sort([(blockrowind[i],blockcolind[i]) for i in 1:length(blockrowind)],by=x->x[1])
  sort!(sortedinds,by=x->x[2],alg=InsertionSort)#stable sort keeps the second index in order
  refinds=Array{Int,1}()
  refrowcoords=Array{Int,1}()
  refcolcoords=Array{Int,1}()
  rowheights=cumsum(pushfirst!(copy(rowsizes),1))
  blockheight=0
  blockrow=1
  blockcol=1
  currenti=1
  lastrowind=sortedinds[1][1]-1
  lastcolind=sortedinds[1][2]
  for ind in sortedinds
    rowind,colind=ind
    if colind==lastcolind
      if rowind>lastrowind
        blockheight+=rowsizes[rowind]
      end
    else
      for j in blockcol:blockcol+colsizes[lastcolind]-1
        push!(refinds,currenti)
        push!(refrowcoords,blockrow)
        push!(refcolcoords,j)
        currenti+=blockheight
      end
      blockcol+=colsizes[lastcolind]
      blockrow=rowheights[rowind]
      blockheight=rowsizes[rowind]
    end
    lastcolind=colind
    lastrowind=rowind
  end
  for j in blockcol:blockcol+colsizes[lastcolind]-1
    push!(refinds,currenti)
    push!(refrowcoords,blockrow)
    push!(refcolcoords,j)
    currenti+=blockheight
  end
  push!(refinds,currenti)#guard
  push!(refrowcoords,-1)
  push!(refcolcoords,-1)
  rowindobj=BlockBandedMatrixIndex(currenti-1,refinds,refrowcoords,true)
  colindobj=BlockBandedMatrixIndex(currenti-1,refinds,refcolcoords,false)
  rowindobj,colindobj
end

struct BandedBlockBandedMatrixIndex <: MatrixIndex
  count::Int
  refinds::Array{Int,1}
  refcoords::Array{Int,1}#storing col or row inds at ref points
  reflocalinds::Array{BandedMatrixIndex,1}
  isrow::Bool
end

function BandedBlockBandedMatrixIndex(nrowblock,ncolblock,rowsizes,colsizes,l,u,lambda,mu)
  blockrowind=BandedMatrixIndex(nrowblock,ncolblock,l,u,true)
  blockcolind=BandedMatrixIndex(nrowblock,ncolblock,l,u,false)
  sortedinds=sort([(blockrowind[i],blockcolind[i]) for i in 1:length(blockrowind)],by=x->x[1])
  sort!(sortedinds,by=x->x[2],alg=InsertionSort)#stable sort keeps the second index in order
  rowheights=cumsum(pushfirst!(copy(rowsizes),1))
  colwidths=cumsum(pushfirst!(copy(colsizes),1))
  currenti=1
  refinds=Array{Int,1}()
  refrowcoords=Array{Int,1}()
  refcolcoords=Array{Int,1}()
  reflocalrowinds=Array{BandedMatrixIndex,1}()
  reflocalcolinds=Array{BandedMatrixIndex,1}()
  for ind in sortedinds
    rowind,colind=ind
    localrowind=BandedMatrixIndex(rowsizes[rowind],colsizes[colind],lambda,mu,true)
    localcolind=BandedMatrixIndex(rowsizes[rowind],colsizes[colind],lambda,mu,false)
    push!(refinds,currenti)
    push!(refrowcoords,rowheights[rowind])
    push!(refcolcoords,colwidths[colind])
    push!(reflocalrowinds,localrowind)
    push!(reflocalcolinds,localcolind)
    currenti+=localrowind.count
  end
  push!(refinds,currenti)
  push!(refrowcoords,-1)
  push!(refcolcoords,-1)
  rowindobj=BandedBlockBandedMatrixIndex(currenti-1,refinds,refrowcoords,reflocalrowinds,true)
  colindobj=BandedBlockBandedMatrixIndex(currenti-1,refinds,refcolcoords,reflocalcolinds,false)
  rowindobj,colindobj
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

function Base.getindex(ind::BandedMatrixIndex,i::Int)
  1 <= i <= ind.count || throw(BoundsError(ind, i))
  _i=i
  p=1
  while _i-ind.bandsizes[p]>0
    _i-=ind.bandsizes[p]
    p+=1
  end
  bandind=ind.bandinds[p]
  startfromone=!xor(ind.isrow,(bandind>0))
  if startfromone
    return _i
  else
    return _i+abs(bandind)
  end
end

function Base.getindex(ind::BlockBandedMatrixIndex,i::Int)
  1 <= i <= ind.count || throw(BoundsError(ind, i))
  p=1
  while i-ind.refinds[p]>=0
    p+=1
  end
  p-=1
  _i=i-ind.refinds[p]
  if ind.isrow
    return ind.refcoords[p]+_i
  else
    return ind.refcoords[p]
  end
end

function Base.getindex(ind::BandedBlockBandedMatrixIndex,i::Int)
  1 <= i <= ind.count || throw(BoundsError(ind, i))
  p=1
  while i-ind.refinds[p]>=0
    p+=1
  end
  p-=1
  _i=i-ind.refinds[p]+1
  ind.reflocalinds[p][_i]+ind.refcoords[p]-1
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

"""
  lu_instance(A) -> lu_factorization_instance

Return an instance of the LU factorization object with the correct type
cheaply.
"""
function lu_instance(A::Matrix{T}) where T
  noUnitT = typeof(zero(T))
  luT = LinearAlgebra.lutype(noUnitT)
  ipiv = Vector{LinearAlgebra.BlasInt}(undef, 0)
  info = zero(LinearAlgebra.BlasInt)
  return LU{luT}(similar(A, 0, 0), ipiv, info)
end

"""
  lu_instance(a::Number) -> a

Return the number.
"""
lu_instance(a::Number) = a

"""
safevec(v)

Is a form of `vec` which is safe for all values in vector spaces, i.e. if
is already a vector, like an AbstractVector or Number, it will return said
AbstractVector or Number.
"""
safevec(v) = vec(v)
safevec(v::Number) = v
safevec(v::AbstractVector) = v

"""
zeromatrix(u::AbstractVector)

Creates the zero'd matrix version of `u`. Note that this is unique because
`similar(u,length(u),length(u))` returns a mutable type, so is not type-matching,
while `fill(zero(eltype(u)),length(u),length(u))` doesn't match the array type,
i.e. you'll get a CPU array from a GPU array. The generic fallback is
`u .* u' .* false` which works on a surprising number of types, but can be broken
with weird (recursive) broadcast overloads. For higher order tensors, this
returns the matrix linear operator type which acts on the `vec` of the array.
"""
function zeromatrix(u)
  x = safevec(u)
  x .* x' .* false
end

function __init__()

  @require SuiteSparse="4607b0f0-06f3-5cda-b6b1-a6196a1729e9" begin
    lu_instance(jac_prototype::SparseMatrixCSC) = SuiteSparse.UMFPACK.UmfpackLU(Ptr{Cvoid}(), Ptr{Cvoid}(), 1, 1,
                                                                                      jac_prototype.colptr[1:1],
                                                                                      jac_prototype.rowval[1:1],
                                                                                      jac_prototype.nzval[1:1],
                                                                                      0)
  end

  @require StaticArrays="90137ffa-7385-5640-81b9-e52037218182" begin
    ismutable(::Type{<:StaticArrays.StaticArray}) = false
    can_setindex(::Type{<:StaticArrays.StaticArray}) = false
    ismutable(::Type{<:StaticArrays.MArray}) = true
    function lu_instance(_A::StaticArrays.StaticMatrix{N,N}) where {N}
      A = StaticArrays.SArray(_A)
      L = LowerTriangular(A)
      U = UpperTriangular(A)
      p = StaticArrays.SVector{N,Int}(1:N)
      return StaticArrays.LU(L, U, p)
    end
  end

  @require LabelledArrays="2ee39098-c373-598a-b85f-a56591580800" begin
    ismutable(::Type{<:LabelledArrays.LArray{T,N,Syms}}) where {T,N,Syms} = ismutable(T)
    can_setindex(::Type{<:LabelledArrays.SLArray}) = false
  end

  @require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
    ismutable(::Type{<:Tracker.TrackedArray}) = false
    can_setindex(::Type{<:Tracker.TrackedArray}) = false
    fast_scalar_indexing(::Type{<:Tracker.TrackedArray}) = false
    aos_to_soa(x::AbstractArray{<:Tracker.TrackedReal,N}) where N = Tracker.collect(x)
  end

  @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    include("cuarrays.jl")
  end

  @require BandedMatrices="aae01518-5342-5314-be14-df237901396f" begin
    function findstructralnz(x::BandedMatrices.BandedMatrix)
      l,u=BandedMatrices.bandwidths(x)
      rowsize,colsize=size(x)
      rowind=BandedMatrixIndex(rowsize,colsize,l,u,true)
      colind=BandedMatrixIndex(rowsize,colsize,l,u,false)
      (rowind,colind)
    end

    has_sparsestruct(::Type{<:BandedMatrices.BandedMatrix}) = true
    is_structured(::Type{<:BandedMatrices.BandedMatrix}) = true
    fast_matrix_colors(::Type{<:BandedMatrices.BandedMatrix}) = true

    function matrix_colors(A::BandedMatrices.BandedMatrix)
        l,u=BandedMatrices.bandwidths(A)
        width=u+l+1
        _cycle(1:width,size(A,2))
    end

  end

  @require BlockBandedMatrices="ffab5731-97b5-5995-9138-79e8c1846df0" begin
    @require BlockArrays="8e7c35d0-a365-5155-bbbb-fb81a777f24e" begin
      function findstructralnz(x::BlockBandedMatrices.BlockBandedMatrix)
        l,u=BlockBandedMatrices.blockbandwidths(x)
        nrowblock=BlockBandedMatrices.blocksize(x,1)
        ncolblock=BlockBandedMatrices.blocksize(x,2)
        rowsizes=BlockArrays.blocklengths(axes(x,1))
        colsizes=BlockArrays.blocklengths(axes(x,2))
        BlockBandedMatrixIndex(nrowblock,ncolblock,rowsizes,colsizes,l,u)
      end

      function findstructralnz(x::BlockBandedMatrices.BandedBlockBandedMatrix)
        l,u=BlockBandedMatrices.blockbandwidths(x)
        lambda,mu=BlockBandedMatrices.subblockbandwidths(x)
        nrowblock=BlockBandedMatrices.blocksize(x,1)
        ncolblock=BlockBandedMatrices.blocksize(x,2)
        rowsizes=BlockArrays.blocklengths(axes(x,1))
        colsizes=BlockArrays.blocklengths(axes(x,2))
        BandedBlockBandedMatrixIndex(nrowblock,ncolblock,rowsizes,colsizes,l,u,lambda,mu)
      end

      has_sparsestruct(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
      has_sparsestruct(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true
      is_structured(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
      is_structured(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true
      fast_matrix_colors(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
      fast_matrix_colors(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true

      function matrix_colors(A::BlockBandedMatrices.BlockBandedMatrix)
          l,u=BlockBandedMatrices.blockbandwidths(A)
          blockwidth=l+u+1
          nblock=BlockBandedMatrices.blocksize(A,2)
          cols=BlockArrays.blocklengths(axes(A,2))
          blockcolors=_cycle(1:blockwidth,nblock)
          #the reserved number of colors of a block is the maximum length of columns of blocks with the same block color
          ncolors=[maximum(cols[i:blockwidth:nblock]) for i in 1:blockwidth]
          endinds=cumsum(ncolors)
          startinds=[endinds[i]-ncolors[i]+1 for i in 1:blockwidth]
          colors=[(startinds[blockcolors[i]]:endinds[blockcolors[i]])[1:cols[i]] for i in 1:nblock]
          vcat(colors...)
      end

      function matrix_colors(A::BlockBandedMatrices.BandedBlockBandedMatrix)
          l,u=BlockBandedMatrices.blockbandwidths(A)
          lambda,mu=BlockBandedMatrices.subblockbandwidths(A)
          blockwidth=l+u+1
          subblockwidth=lambda+mu+1
          nblock=BlockBandedMatrices.blocksize(A,2)
          cols=BlockArrays.blocklengths(axes(A,2))
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

end
