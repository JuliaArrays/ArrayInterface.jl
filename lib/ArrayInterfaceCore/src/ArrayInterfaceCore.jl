module ArrayInterfaceCore

using LinearAlgebra
using LinearAlgebra: AbstractTriangular
using SparseArrays
using SuiteSparse

@static if isdefined(Base, :ReshapedReinterpretArray)
    _is_reshaped(::Type{<:Base.ReshapedReinterpretArray}) = true
end
_is_reshaped(::Type{<:Base.ReinterpretArray}) = false

@static if isdefined(Base, Symbol("@assume_effects"))
    using Base: @assume_effects
else
    macro assume_effects(_, ex)
        Base.@pure ex
    end
end

@assume_effects :total __parameterless_type(T) = Base.typename(T).wrapper
parameterless_type(x) = parameterless_type(typeof(x))
parameterless_type(x::Type) = __parameterless_type(x)

const VecAdjTrans{T,V<:AbstractVector{T}} = Union{Transpose{T,V},Adjoint{T,V}}
const MatAdjTrans{T,M<:AbstractMatrix{T}} = Union{Transpose{T,M},Adjoint{T,M}}
const UpTri{T,M} = Union{UpperTriangular{T,M},UnitUpperTriangular{T,M}}
const LoTri{T,M} = Union{LowerTriangular{T,M},UnitLowerTriangular{T,M}}

"""
    parent_type(::Type{T}) -> Type

Returns the parent array that type `T` wraps.
"""
parent_type(x) = parent_type(typeof(x))
parent_type(::Type{<:SubArray{T,N,P}}) where {T,N,P} = P
parent_type(::Type{<:Base.ReshapedArray{T,N,P}}) where {T,N,P} = P
parent_type(::Type{Adjoint{T,S}}) where {T,S} = S
parent_type(::Type{Transpose{T,S}}) where {T,S} = S
parent_type(::Type{Symmetric{T,S}}) where {T,S} = S
parent_type(::Type{<:AbstractTriangular{T,S}}) where {T,S} = S
parent_type(::Type{<:PermutedDimsArray{T,N,I1,I2,A}}) where {T,N,I1,I2,A} = A
parent_type(::Type{Base.Slice{T}}) where {T} = T
parent_type(::Type{T}) where {T} = T
parent_type(::Type{R}) where {S,T,A,N,R<:Base.ReinterpretArray{T,N,S,A}} = A
parent_type(::Type{Diagonal{T,V}}) where {T,V} = V

"""
    buffer(x)

Return the buffer data that `x` points to. Unlike `parent(x::AbstractArray)`, `buffer(x)`
may not return another array type.
"""
buffer(x) = parent(x)
buffer(x::SparseMatrixCSC) = getfield(x, :nzval)
buffer(x::SparseVector) = getfield(x, :nzval)

"""
    can_change_size(::Type{T}) -> Bool

Returns `true` if the Base.size of `T` can change, in which case operations
such as `pop!` and `popfirst!` are available for collections of type `T`.
"""
can_change_size(x) = can_change_size(typeof(x))
can_change_size(::Type{T}) where {T} = false
can_change_size(::Type{<:Vector}) = true
can_change_size(::Type{<:AbstractDict}) = true
can_change_size(::Type{<:Base.ImmutableDict}) = false

function ismutable end

"""
    ismutable(::Type{T}) -> Bool

Query whether instances of type `T` are mutable or not, see
https://github.com/JuliaDiffEq/RecursiveArrayTools.jl/issues/19.
"""
ismutable(x) = ismutable(typeof(x))
function ismutable(::Type{T}) where {T<:AbstractArray}
    if parent_type(T) <: T
        return true
    else
        return ismutable(parent_type(T))
    end
end
ismutable(::Type{<:AbstractRange}) = false
ismutable(::Type{<:AbstractDict}) = true
ismutable(::Type{<:Base.ImmutableDict}) = false
ismutable(::Type{BigFloat}) = false
ismutable(::Type{BigInt}) = false
function ismutable(::Type{T}) where {T}
    if parent_type(T) <: T
        @static if VERSION ≥ v"1.7.0-DEV.1208"
            return Base.ismutabletype(T)
        else
            return T.mutable
        end
    else
        return ismutable(parent_type(T))
    end
end

# Piracy
function Base.setindex(x::AbstractArray, v, i...)
    _x = Base.copymutable(x)
    _x[i...] = v
    return _x
end

function Base.setindex(x::AbstractVector, v, i::Int)
    n = length(x)
    x .* (i .!== 1:n) .+ v .* (i .== 1:n)
end

function Base.setindex(x::AbstractMatrix, v, i::Int, j::Int)
    n, m = Base.size(x)
    x .* (i .!== 1:n) .* (j .!== i:m)' .+ v .* (i .== 1:n) .* (j .== i:m)'
end

"""
    can_setindex(::Type{T}) -> Bool

Query whether a type can use `setindex!`.
"""
can_setindex(x) = can_setindex(typeof(x))
can_setindex(::Type) = true
can_setindex(::Type{<:AbstractRange}) = false
can_setindex(::Type{<:AbstractDict}) = true
can_setindex(::Type{<:Base.ImmutableDict}) = false
can_setindex(@nospecialize T::Type{<:Tuple}) = false
can_setindex(@nospecialize T::Type{<:NamedTuple}) = false
can_setindex(::Type{<:Base.Iterators.Pairs{<:Any,<:Any,P}}) where {P} = can_setindex(P)

"""
    aos_to_soa(x)

Converts an array of structs formulation to a struct of array.
"""
aos_to_soa(x) = x

"""
    isstructured(::Type{T}) -> Bool

Query whether a type is a representation of a structured matrix.
"""
isstructured(x) = isstructured(typeof(x))
isstructured(::Type) = false
isstructured(::Type{<:Symmetric}) = true
isstructured(::Type{<:Hermitian}) = true
isstructured(::Type{<:UpperTriangular}) = true
isstructured(::Type{<:LowerTriangular}) = true
isstructured(::Type{<:Tridiagonal}) = true
isstructured(::Type{<:SymTridiagonal}) = true
isstructured(::Type{<:Bidiagonal}) = true
isstructured(::Type{<:Diagonal}) = true

"""
    has_sparsestruct(x::AbstractArray) -> Bool

Determine whether `findstructralnz` accepts the parameter `x`.
"""
has_sparsestruct(x) = has_sparsestruct(typeof(x))
has_sparsestruct(::Type) = false
has_sparsestruct(::Type{<:AbstractArray}) = false
has_sparsestruct(::Type{<:SparseMatrixCSC}) = true
has_sparsestruct(::Type{<:Diagonal}) = true
has_sparsestruct(::Type{<:Bidiagonal}) = true
has_sparsestruct(::Type{<:Tridiagonal}) = true
has_sparsestruct(::Type{<:SymTridiagonal}) = true

"""
    issingular(A::AbstractMatrix) -> Bool

Determine whether a given abstract matrix is singular.
"""
issingular(A::AbstractMatrix) = issingular(Matrix(A))
issingular(A::AbstractSparseMatrix) = !issuccess(lu(A, check=false))
issingular(A::Matrix) = !issuccess(lu(A, check=false))
issingular(A::UniformScaling) = A.λ == 0
issingular(A::Diagonal) = any(iszero, A.diag)
issingular(A::Bidiagonal) = any(iszero, A.dv)
issingular(A::SymTridiagonal) = diaganyzero(ldlt(A).data)
issingular(A::Tridiagonal) = !issuccess(lu(A, check=false))
issingular(A::Union{Hermitian,Symmetric}) = diaganyzero(bunchkaufman(A, check=false).LD)
issingular(A::Union{LowerTriangular,UpperTriangular}) = diaganyzero(A.data)
issingular(A::Union{UnitLowerTriangular,UnitUpperTriangular}) = false
issingular(A::Union{Adjoint,Transpose}) = issingular(parent(A))
diaganyzero(A) = any(iszero, view(A, diagind(A)))

"""
    findstructralnz(x::AbstractArray)

Return: (I,J) #indexable objects
Find sparsity pattern of special matrices, the same as the first two elements of findnz(::SparseMatrixCSC).
"""
function findstructralnz(x::Diagonal)
    n = Base.size(x, 1)
    (1:n, 1:n)
end

function findstructralnz(x::Bidiagonal)
    n = Base.size(x, 1)
    isup = x.uplo == 'U' ? true : false
    rowind = BidiagonalIndex(n + n - 1, isup)
    colind = BidiagonalIndex(n + n - 1, !isup)
    (rowind, colind)
end

function findstructralnz(x::Union{Tridiagonal,SymTridiagonal})
    n = Base.size(x, 1)
    rowind = TridiagonalIndex(n + n - 1 + n - 1, n, true)
    colind = TridiagonalIndex(n + n - 1 + n - 1, n, false)
    (rowind, colind)
end

function findstructralnz(x::SparseMatrixCSC)
    rowind, colind, _ = findnz(x)
    (rowind, colind)
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
`[1,2,3,..., Base.size(A,2)]`.
"""
function matrix_colors(A::Union{Array,UpperTriangular,LowerTriangular})
    eachindex(1:Base.size(A, 2)) # Vector Base.size matches number of rows
end
matrix_colors(A::Diagonal) = fill(1, Base.size(A, 2))
matrix_colors(A::Bidiagonal) = _cycle(1:2, Base.size(A, 2))
matrix_colors(A::Union{Tridiagonal,SymTridiagonal}) = _cycle(1:3, Base.size(A, 2))
_cycle(repetend, len) = repeat(repetend, div(len, length(repetend)) + 1)[1:len]

"""
  lu_instance(A) -> lu_factorization_instance

Returns an instance of the LU factorization object with the correct type
cheaply.
"""
function lu_instance(A::Matrix{T}) where {T}
    noUnitT = typeof(zero(T))
    luT = LinearAlgebra.lutype(noUnitT)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, 0)
    info = zero(LinearAlgebra.BlasInt)
    return LU{luT}(similar(A, 0, 0), ipiv, info)
end
function lu_instance(jac_prototype::SparseMatrixCSC)
    SuiteSparse.UMFPACK.UmfpackLU(
        Ptr{Cvoid}(),
        Ptr{Cvoid}(),
        1,
        1,
        jac_prototype.colptr[1:1],
        jac_prototype.rowval[1:1],
        jac_prototype.nzval[1:1],
        0,
    )
end

"""
  lu_instance(a::Number) -> a

Returns the number.
"""
lu_instance(a::Number) = a

"""
    lu_instance(a::Any) -> lu(a, check=false)

Returns the number.
"""
lu_instance(a::Any) = lu(a, check=false)

"""
    safevec(v)

It is a form of `vec` which is safe for all values in vector spaces, i.e., if it
is already a vector, like an AbstractVector or Number, it will return said
AbstractVector or Number.
"""
safevec(v) = vec(v)
safevec(v::Number) = v
safevec(v::AbstractVector) = v

"""
    zeromatrix(u::AbstractVector)

Creates the zero'd matrix version of `u`. Note that this is unique because
`similar(u,length(u),length(u))` returns a mutable type, so it is not type-matching,
while `fill(zero(eltype(u)),length(u),length(u))` doesn't match the array type,
i.e., you'll get a CPU array from a GPU array. The generic fallback is
`u .* u' .* false`, which works on a surprising number of types, but can be broken
with weird (recursive) broadcast overloads. For higher-order tensors, this
returns the matrix linear operator type which acts on the `vec` of the array.
"""
function zeromatrix(u)
    x = safevec(u)
    x .* x' .* false
end

# Reduces compile time burdens
function zeromatrix(u::Array{T}) where {T}
    out = Matrix{T}(undef, length(u), length(u))
    fill!(out, false)
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
function restructure(x, y)
    out = similar(x, eltype(y))
    vec(out) .= vec(y)
    out
end

function restructure(x::Array, y)
    reshape(convert(Array, y), Base.size(x)...)
end

abstract type AbstractDevice end
abstract type AbstractCPU <: AbstractDevice end
struct CPUPointer <: AbstractCPU end
struct CPUTuple <: AbstractCPU end
struct CheckParent end
struct CPUIndex <: AbstractCPU end
struct GPU <: AbstractDevice end

"""
    can_avx(f) -> Bool

Returns `true` if the function `f` is guaranteed to be compatible with
`LoopVectorization.@avx` for supported element and array types. While a return
value of `false` does not indicate the function isn't supported, this allows a
library to conservatively apply `@avx` only when it is known to be safe to do so.

```julia
function mymap!(f, y, args...)
    if can_avx(f)
        @avx @. y = f(args...)
    else
        @. y = f(args...)
    end
end
```
"""
can_avx(::Any) = false

"""
    fast_scalar_indexing(::Type{T}) -> Bool

Query whether an array type has fast scalar indexing.
"""
fast_scalar_indexing(x) = fast_scalar_indexing(typeof(x))
fast_scalar_indexing(::Type) = true
fast_scalar_indexing(::Type{<:LinearAlgebra.AbstractQ}) = false
fast_scalar_indexing(::Type{<:LinearAlgebra.LQPackedQ}) = false

"""
    allowed_getindex(x,i...)

A scalar `getindex` which is always allowed.
"""
allowed_getindex(x, i...) = x[i...]

"""
    allowed_setindex!(x,v,i...)

A scalar `setindex!` which is always allowed.
"""
allowed_setindex!(x, v, i...) = Base.setindex!(x, v, i...)

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

Base.firstindex(i::Union{BidiagonalIndex,TridiagonalIndex}) = 1
Base.lastindex(i::Union{BidiagonalIndex,TridiagonalIndex}) = i.count
Base.length(i::Union{BidiagonalIndex,TridiagonalIndex}) = lastindex(i)

Base.@propagate_inbounds function Base.getindex(ind::BidiagonalIndex, i::Int)
    @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
    if ind.isup
        ii = i + 1
    else
        ii = i + 1 + 1
    end
    convert(Int, floor(ii / 2))
end

Base.@propagate_inbounds function Base.getindex(ind::TridiagonalIndex, i::Int)
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

_cartesian_index(i::Tuple{Vararg{Int}}) = CartesianIndex(i)
_cartesian_index(::Any) = nothing

"""
    known_first(::Type{T}) -> Union{Int,Nothing}

If `first` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

```julia
julia> ArrayInterface.known_first(typeof(1:4))
nothing

julia> ArrayInterface.known_first(typeof(Base.OneTo(4)))
1
```
"""
known_first(x) = known_first(typeof(x))
function known_first(::Type{T}) where {T}
    if parent_type(T) <: T
        return nothing
    else
        return known_first(parent_type(T))
    end
end
known_first(::Type{Base.OneTo{T}}) where {T} = 1
known_first(::Type{<:Base.IdentityUnitRange{T}}) where {T} = known_first(T)
function known_first(::Type{<:CartesianIndices{N,R}}) where {N,R}
    _cartesian_index(ntuple(i -> known_first(R.parameters[i]), Val(N)))
end

"""
    known_last(::Type{T}) -> Union{Int,Nothing}

If `last` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

```julia
julia> ArrayInterface.known_last(typeof(1:4))
nothing

julia> ArrayInterface.known_first(typeof(static(1):static(4)))
4

```
"""
known_last(x) = known_last(typeof(x))
known_last(::Type{T}) where {T} = parent_type(T) <: T ? nothing : known_last(parent_type(T))
function known_last(::Type{<:CartesianIndices{N,R}}) where {N,R}
    _cartesian_index(ntuple(i -> known_last(R.parameters[i]), Val(N)))
end

"""
    known_step(::Type{T}) -> Union{Int,Nothing}

If `step` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

```julia
julia> ArrayInterface.known_step(typeof(1:2:8))
nothing

julia> ArrayInterface.known_step(typeof(1:4))
1

```
"""
known_step(x) = known_step(typeof(x))
known_step(::Type{T}) where {T} = parent_type(T) <: T ? nothing : known_step(parent_type(T))
known_step(@nospecialize T::Type{<:AbstractUnitRange}) = 1

"""
    is_splat_index(::Type{T}) -> Bool
Returns `static(true)` if `T` is a type that splats across multiple dimensions. 
"""
is_splat_index(T::Type) = false
is_splat_index(@nospecialize(x)) = is_splat_index(typeof(x))

"""
    ndims_index(::Type{I}) -> Int

Returns the number of dimension that an instance of `I` maps to when indexing. For example,
`CartesianIndex{3}` maps to 3 dimensions. If this method is not explicitly defined, then `1`
is returned.
"""
ndims_index(::Type{<:Base.AbstractCartesianIndex{N}}) where {N} = N
# preserve CartesianIndices{0} as they consume a dimension.
ndims_index(::Type{CartesianIndices{0,Tuple{}}}) = 1
ndims_index(@nospecialize T::Type{<:AbstractArray{Bool}}) = ndims(T)
ndims_index(@nospecialize T::Type{<:AbstractArray}) = ndims_index(eltype(T))
ndims_index(@nospecialize T::Type{<:Base.LogicalIndex}) = ndims(fieldtype(T, :mask))
ndims_index(T::DataType) = 1
ndims_index(@nospecialize(i)) = ndims_index(typeof(i))

end # module
