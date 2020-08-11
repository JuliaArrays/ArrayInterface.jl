module ArrayInterface

using Requires
using LinearAlgebra
using SparseArrays

Base.@pure __parameterless_type(T) = Base.typename(T).wrapper
parameterless_type(x) = parameterless_type(typeof(x))
parameterless_type(x::Type) = __parameterless_type(x)

"""
    parent_type(x)

Returns the parent array that `x` wraps.
"""
parent_type(x) = parent_type(typeof(x))
parent_type(::Type{<:SubArray{T,N,P}}) where {T,N,P} = P
parent_type(::Type{<:Base.ReshapedArray{T,N,P}}) where {T,N,P} = P
parent_type(::Type{Adjoint{T,S}}) where {T,S} = S
parent_type(::Type{Transpose{T,S}}) where {T,S} = S
parent_type(::Type{Symmetric{T,S}}) where {T,S} = S
parent_type(::Type{<:LinearAlgebra.AbstractTriangular{T,S}}) where {T,S} = S
parent_type(::Type{T}) where {T} = T

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
    lu_instance(a::Any) -> lu(a, check=false)

Return the number.
"""
lu_instance(a::Any) = lu(a, check=false)

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

"""
restructure(x,y)

Restructures the object `y` into a shape of `x`, keeping its values intact. For
simple objects like an `Array`, this simply amounts to a reshape. However, for
more complex objects such as an `ArrayPartition`, not all of the structural
information is adequately contained in the type for standard tools to work. In
these cases, `restructure` gives a way to convert for example an `Array` into
a matching `ArrayPartition`.
"""
function restructure(x,y)
  out = similar(x,eltype(y))
  vec(out) .= vec(y)
end

function restructure(x::Array,y)
  reshape(convert(Array,y),size(x)...)
end

"""
known_first(::Type{T})

If `first` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

@test isnothing(known_first(typeof(1:4)))
@test isone(known_first(typeof(Base.OneTo(4))))
"""
known_first(x) = known_first(typeof(x))
known_first(::Type{T}) where {T} = nothing
known_first(::Type{Base.OneTo{T}}) where {T} = one(T)

"""
known_last(::Type{T})

If `last` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

@test isnothing(known_last(typeof(1:4)))
using StaticArrays
@test known_last(typeof(SOneTo(4))) == 4
"""
known_last(x) = known_last(typeof(x))
known_last(::Type{T}) where {T} = nothing

"""
known_step(::Type{T})

If `step` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

@test isnothing(known_step(typeof(1:0.2:4)))
@test isone(known_step(typeof(1:4)))
"""
known_step(x) = known_step(typeof(x))
known_step(::Type{T}) where {T} = nothing
known_step(::Type{<:AbstractUnitRange{T}}) where {T} = one(T)

"""
is_cpu_column_major(::Type{T})

Does an Array of type `T` point to column major memory in the cpu's address space?
If `is_cpu_column_major(typeof(A))` return `true` and the element type is a primite
type, then the array should be compatible with `LoopVectorization.jl` as well as
`C` and `Fortran` programs requiring pointers and assuming column major memory layout.

If `is_cpu_column_major(typeof(A))` return `true`, the array supports the
[Strided Array](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-strided-arrays-1) interface.

"""
is_cpu_column_major(x) = is_cpu_column_major(typeof(x))
is_cpu_column_major(::Type) = false
is_cpu_column_major(::Type{<:Array}) = true
is_cpu_column_major(::Type{S}) where {A, S <: SubArray{<:Any,<:Any,A,<:Tuple{Vararg{Union{Int,<:AbstractRange}}}}} = is_cpu_column_major(A)

"""
stridelayout(::Type{T}) -> (contig, batch, striderank)

Descrive the memory layout of a strided container of type `T`. If unknown or not strided, returns `nothing`.
Else, it returns a tuple with elements:
 - `contig`: The axis with contiguous elements. `contig == -1` indicates no axis is contiguous. `striderank[contig]` does not necessarilly equal `1`.
 - `batch`: indicates the number of contiguous elements. That is, if `batch == 16`, then axis `contig` will contain batches of 16 contiguous elements interleaved with axis `findfirst(isone.(striderank))`.
 - `striderank` indicates the rank of the given stride with respect to the others. If for `A::T` we have `striderank[i] > striderank[j]`, then `stride(A,i) > stride(A,j)`.

The convenience method
```julia
stridelayout(x) = stridelayout(typeof(x))
```
is also provided.

```julia
julia> A = rand(3,4,5);

julia> stridelayout(A)
 (1, 1, Base.OneTo(3))

julia> stridelayout(PermutedDimsArray(A,(3,1,2)))
 (2, 1, (3, 1, 2))

julia> stridelayout(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))
 (1, 1, (1, 2))

julia> stridelayout(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))
 (2, 1, (3, 1, 2))

julia> stridelayout(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))
 (-1, 1, (2, 1))
```
"""
stridelayout(x) = stridelayout(typeof(x))
stridelayout(::Type) = nothing
stridelayout(::Type{Array{T,N}}) where {T,N} = (1,1,Base.OneTo(N))
stridelayout(::Type{<:Tuple}) = (1,1,Base.OneTo(1))
function stridelayout(::Type{<:Union{Transpose{T,A},Adjoint{T,A}}}) where {T,A<:AbstractMatrix{T}}
    ml = stridelayout(A)
    isnothing(ml) && return nothing
    contig, batch, rank = ml
    new_rank = (rank[2], rank[1])
    new_contig = congig == -1 ? -1 : 3 - contig
    new_contig, batch, new_rank
end
function stridelayout(::Type{<:PermutedDimsArray{T,N,I1,I2,A}}) where {T,N,I1,I2,A<:AbstractArray{T,N}}
    ml = stridelayout(A)
    isnothing(ml) && return nothing
    contig, batch, rank = ml
    new_contig = I2[contig]
    new_rank = ntuple(n -> rank[I1[n]], Val(N))
    new_contig, batch, new_rank
end
@generated function stridelayout(::Type{S}) where {N,NP,T,A<:AbstractArray{T,NP},I,S <: SubArray{T,N,A,I}}
    ml = stridelayout(A)
    isnothing(ml) && return nothing
    contig, batch, rank = ml
    rankv = collect(rank)
    rank_new = Int[]
    n = 0
    new_contig = contig
    for np in 1:NP
        r = rankv[np]
        if I.parameters[np] <: AbstractUnitRange
            n += 1
            push!(rank_new, r)
            if np == contig
                new_contig = n
            end
        else
            # There's definitely a smarter way to do this.
            # When we drop a rank, we lower the others.
            for nᵢ ∈ 1:n
                rᵢ = rank_new[nᵢ]
                if rᵢ > r
                    rank_new[nᵢ] = rᵢ - 1
                end
            end
            for npᵢ ∈ np+1:NP
                rᵢ = rankv[npᵢ]
                if rᵢ > r
                    rankv[npᵢ] = rᵢ - 1
                end
            end
            if np == contig
                new_contig = -1
            end
        end
    end
    # If n != N, then an axis was indeced by something other than an integer or `AbstractUnitRange`, so we return `nothing`
    n == N || return nothing
    ranktup = Expr(:tuple); append!(ranktup.args, rank_new) # dynamic splats bad
    Expr(:tuple, new_contig, batch, ranktup)
end

"""
canavx(f)

Returns `true` if the function `f` is guaranteed to be compatible with `LoopVectorization.@avx` for supported element and array types.
While a return value of `false` does not indicate the function isn't supported, this allows a library to conservatively apply `@avx`
only when it is known to be safe to do so.

```julia
function mymap!(f, y, args...)
    if canavx(f)
        @avx @. y = f(args...)
    else
        @. y = f(args...)
    end
end
```
"""
canavx(::Any) = false


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
    ismutable(::Type{<:StaticArrays.SizedArray}) = true

    function lu_instance(_A::StaticArrays.StaticMatrix{N,N}) where {N}
      A = StaticArrays.SArray(_A)
      L = LowerTriangular(A)
      U = UpperTriangular(A)
      p = StaticArrays.SVector{N,Int}(1:N)
      return StaticArrays.LU(L, U, p)
    end

    function restructure(x::StaticArrays.SArray,y::StaticArrays.SArray)
      reshape(y,StaticArrays.Size(x))
    end

    function restructure(x::StaticArrays.SArray{S},y) where S
      StaticArrays.SArray{S}(y)
    end

    known_first(::Type{<:StaticArrays.SOneTo}) = 1
    known_last(::Type{StaticArrays.SOneTo{N}}) where {N} = N

    is_cpu_column_major(::Type{<:StaticArrays.MArray}) = true
    # is_cpu_column_major(::Type{<:StaticArrays.SizedArray}) = false # Why?
    stridelayout(::Type{<:StaticArrays.StaticArray{S,T,N}}) where {S,T,N} = (1,1,Base.OneTo(N))

    @require Adapt="79e6a3ab-5dfb-504d-930d-738a2a938a0e" begin
      function Adapt.adapt_storage(::Type{<:StaticArrays.SArray{S}},xs::Array) where S
          StaticArrays.SArray{S}(xs)
      end
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
    @require Adapt="79e6a3ab-5dfb-504d-930d-738a2a938a0e" begin
      include("cuarrays.jl")
    end
    @require DiffEqBase="2b5f629d-d688-5b77-993f-72d75c75574e" begin
      # actually do QR
      lu_instance(A::CuArrays.CuMatrix{T}) where T = CuArrays.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
    end
  end

  @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
    @require Adapt="79e6a3ab-5dfb-504d-930d-738a2a938a0e" begin
      include("cuarrays2.jl")
    end
    @require DiffEqBase="2b5f629d-d688-5b77-993f-72d75c75574e" begin
      # actually do QR
      lu_instance(A::CUDA.CuMatrix{T}) where T = CUDA.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
    end
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
