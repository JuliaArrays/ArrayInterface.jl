module ArrayInterface

using IfElse
using Requires
using LinearAlgebra
using SparseArrays
using Static
using Static: Zero, One, nstatic, eq, ne, gt, ge, lt, le, eachop, eachop_tuple,
    find_first_eq, permute, invariant_permutation, field_type, reduce_tup
using Base.Cartesian
import Compat

using Base: @propagate_inbounds, tail, OneTo, LogicalIndex, Slice, ReinterpretArray,
    ReshapedArray, AbstractCartesianIndex

const CanonicalInt = Union{Int,StaticInt}
canonicalize(x::Integer) = Int(x)
canonicalize(@nospecialize(x::StaticInt)) = x

@static if isdefined(Base, :ReshapedReinterpretArray)
    _is_reshaped(::Type{<:Base.ReshapedReinterpretArray}) = true
end
_is_reshaped(::Type{<:ReinterpretArray}) = false

@generated function merge_tuple_type(::Type{X}, ::Type{Y}) where {X<:Tuple,Y<:Tuple}
    Tuple{X.parameters..., Y.parameters...}
end
Base.@pure __parameterless_type(T) = Base.typename(T).wrapper
parameterless_type(x) = parameterless_type(typeof(x))
parameterless_type(x::Type) = __parameterless_type(x)

const VecAdjTrans{T,V<:AbstractVector{T}} = Union{Transpose{T,V},Adjoint{T,V}}
const MatAdjTrans{T,M<:AbstractMatrix{T}} = Union{Transpose{T,M},Adjoint{T,M}}
const UpTri{T,M} = Union{UpperTriangular{T,M},UnitUpperTriangular{T,M}}
const LoTri{T,M} = Union{LowerTriangular{T,M},UnitLowerTriangular{T,M}}

@inline static_length(a::UnitRange{T}) where {T} = last(a) - first(a) + oneunit(T)
@inline static_length(x) = Static.maybe_static(known_length, length, x)
@inline static_first(x) = Static.maybe_static(known_first, first, x)
@inline static_last(x) = Static.maybe_static(known_last, last, x)
@inline static_step(x) = Static.maybe_static(known_step, step, x)

include("array_index.jl")

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
parent_type(::Type{<:LinearAlgebra.AbstractTriangular{T,S}}) where {T,S} = S
parent_type(::Type{<:PermutedDimsArray{T,N,I1,I2,A}}) where {T,N,I1,I2,A} = A
parent_type(::Type{Slice{T}}) where {T} = T
parent_type(::Type{T}) where {T} = T
parent_type(::Type{R}) where {S,T,A,N,R<:ReinterpretArray{T,N,S,A}} = A
parent_type(::Type{Diagonal{T,V}}) where {T,V} = V

"""
    has_parent(::Type{T}) -> StaticBool

Returns `static(true)` if `parent_type(T)` a type unique to `T`.
"""
has_parent(x) = has_parent(typeof(x))
has_parent(::Type{T}) where {T} = _has_parent(parent_type(T), T)
_has_parent(::Type{T}, ::Type{T}) where {T} = False()
_has_parent(::Type{T1}, ::Type{T2}) where {T1,T2} = True()

"""
    buffer(x)

Return the buffer data that `x` points to. Unlike `parent(x::AbstractArray)`, `buffer(x)`
may not return another array type.
"""
buffer(x) = parent(x)
buffer(x::SparseMatrixCSC) = getfield(x, :nzval)
buffer(x::SparseVector) = getfield(x, :nzval)

"""
    known_length(::Type{T}) -> Union{Int,Missing}

If `length` of an instance of type `T` is known at compile time, return it.
Otherwise, return `missing`.
"""
known_length(x) = known_length(typeof(x))
known_length(::Type{<:NamedTuple{L}}) where {L} = length(L)
known_length(::Type{T}) where {T<:Slice} = known_length(parent_type(T))
known_length(::Type{<:Tuple{Vararg{Any,N}}}) where {N} = N
known_length(::Type{<:Number}) = 1
known_length(::Type{<:AbstractCartesianIndex{N}}) where {N} = N
known_length(::Type{T}) where {T} = _maybe_known_length(Base.IteratorSize(T), T)
_maybe_known_length(::Base.HasShape, ::Type{T}) where {T} = prod(known_size(T))
_maybe_known_length(::Base.IteratorSize, ::Type) = missing
function known_length(::Type{<:Iterators.Flatten{I}}) where {I}
    known_length(I) * known_length(eltype(I))
end

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

"""
    aos_to_soa(x)

Converts an array of structs formulation to a struct of array.
"""
aos_to_soa(x) = x

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
issingular(A::AbstractSparseMatrix) = !issuccess(lu(A, check = false))
issingular(A::Matrix) = !issuccess(lu(A, check = false))
issingular(A::UniformScaling) = A.λ == 0
issingular(A::Diagonal) = any(iszero, A.diag)
issingular(A::Bidiagonal) = any(iszero, A.dv)
issingular(A::SymTridiagonal) = diaganyzero(ldlt(A).data)
issingular(A::Tridiagonal) = !issuccess(lu(A, check = false))
issingular(A::Union{Hermitian,Symmetric}) = diaganyzero(bunchkaufman(A, check = false).LD)
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

function _cycle(repetend, len)
    repeat(repetend, div(len, length(repetend)) + 1)[1:len]
end

function matrix_colors(A::Diagonal)
    fill(1, Base.size(A, 2))
end

function matrix_colors(A::Bidiagonal)
    _cycle(1:2, Base.size(A, 2))
end

function matrix_colors(A::Union{Tridiagonal,SymTridiagonal})
    _cycle(1:3, Base.size(A, 2))
end

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

"""
  lu_instance(a::Number) -> a

Returns the number.
"""
lu_instance(a::Number) = a

"""
    lu_instance(a::Any) -> lu(a, check=false)

Returns the number.
"""
lu_instance(a::Any) = lu(a, check = false)

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
function zeromatrix(u::Array{T}) where T
    out = Matrix{T}(undef, length(u), length(u))
    fill!(out,false)
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
    device(::Type{T}) -> AbstractDevice

Indicates the most efficient way to access elements from the collection in low-level code.
For `GPUArrays`, will return `ArrayInterface.GPU()`.
For `AbstractArray` supporting a `pointer` method, returns `ArrayInterface.CPUPointer()`.
For other `AbstractArray`s and `Tuple`s, returns `ArrayInterface.CPUIndex()`.
Otherwise, returns `missing`.
"""
device(A) = device(typeof(A))
device(::Type) = missing
device(::Type{<:Tuple}) = CPUTuple()
device(::Type{T}) where {T<:Array} = CPUPointer()
device(::Type{T}) where {T<:AbstractArray} = _device(has_parent(T), T)
function _device(::True, ::Type{T}) where {T}
    if defines_strides(T)
        return device(parent_type(T))
    else
        return _not_pointer(device(parent_type(T)))
    end
end
_not_pointer(::CPUPointer) = CPUIndex()
_not_pointer(x) = x
_device(::False, ::Type{T}) where {T<:DenseArray} = CPUPointer()
_device(::False, ::Type{T}) where {T} = CPUIndex()

"""
    defines_strides(::Type{T}) -> Bool

Is strides(::T) defined? It is assumed that types returning `true` also return a valid
pointer on `pointer(::T)`.
"""
defines_strides(x) = defines_strides(typeof(x))
_defines_strides(::Type{T}, ::Type{T}) where {T} = false
_defines_strides(::Type{P}, ::Type{T}) where {P,T} = defines_strides(P)
defines_strides(::Type{T}) where {T} = _defines_strides(parent_type(T), T)
defines_strides(::Type{<:StridedArray}) = true
function defines_strides(::Type{<:SubArray{T,N,P,I}}) where {T,N,P,I}
    return stride_preserving_index(I) === True()
end
defines_strides(::Type{<:BitArray}) = true

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
    insert(collection, index, item)

Returns a new instance of `collection` with `item` inserted into at the given `index`.
"""
Base.@propagate_inbounds function insert(collection, index, item)
    @boundscheck checkbounds(collection, index)
    ret = similar(collection, length(collection) + 1)
    @inbounds for i = firstindex(ret):(index-1)
        ret[i] = collection[i]
    end
    @inbounds ret[index] = item
    @inbounds for i = (index+1):lastindex(ret)
        ret[i] = collection[i-1]
    end
    return ret
end

function insert(x::Tuple{Vararg{Any,N}}, index::Integer, item) where {N}
    @boundscheck if !checkindex(Bool, StaticInt{1}():StaticInt{N}(), index)
        throw(BoundsError(x, index))
    end
    return unsafe_insert(x, Int(index), item)
end

@inline function unsafe_insert(x::Tuple, i::Int, item)
    if i === 1
        return (item, x...)
    else
        return (first(x), unsafe_insert(tail(x), i - 1, item)...)
    end
end

"""
    deleteat(collection, index)

Returns a new instance of `collection` with the item at the given `index` removed.
"""
@propagate_inbounds function deleteat(collection::AbstractVector, index)
    @boundscheck if !checkindex(Bool, eachindex(collection), index)
        throw(BoundsError(collection, index))
    end
    return unsafe_deleteat(collection, index)
end
@propagate_inbounds function deleteat(collection::Tuple{Vararg{Any,N}}, index) where {N}
    @boundscheck if !checkindex(Bool, StaticInt{1}():StaticInt{N}(), index)
        throw(BoundsError(collection, index))
    end
    return unsafe_deleteat(collection, index)
end

function unsafe_deleteat(src::AbstractVector, index::Integer)
    dst = similar(src, length(src) - 1)
    @inbounds for i in indices(dst)
        if i < index
            dst[i] = src[i]
        else
            dst[i] = src[i+1]
        end
    end
    return dst
end

@inline function unsafe_deleteat(src::AbstractVector, inds::AbstractVector)
    dst = similar(src, length(src) - length(inds))
    dst_index = firstindex(dst)
    @inbounds for src_index in indices(src)
        if !in(src_index, inds)
            dst[dst_index] = src[src_index]
            dst_index += one(dst_index)
        end
    end
    return dst
end

@inline function unsafe_deleteat(src::Tuple, inds::AbstractVector)
    dst = Vector{eltype(src)}(undef, length(src) - length(inds))
    dst_index = firstindex(dst)
    @inbounds for src_index in OneTo(length(src))
        if !in(src_index, inds)
            dst[dst_index] = src[src_index]
            dst_index += one(dst_index)
        end
    end
    return Tuple(dst)
end

@inline unsafe_deleteat(x::Tuple{T}, i::Integer) where {T} = ()
@inline unsafe_deleteat(x::Tuple{T1,T2}, i::Integer) where {T1,T2} =
    isone(i) ? (x[2],) : (x[1],)
@inline function unsafe_deleteat(x::Tuple, i::Integer)
    if i === one(i)
        return tail(x)
    elseif i == length(x)
        return Base.front(x)
    else
        return (first(x), unsafe_deleteat(tail(x), i - one(i))...)
    end
end

abstract type AbstractArray2{T,N} <: AbstractArray{T,N} end

Base.size(A::AbstractArray2) = map(Int, ArrayInterface.size(A))
Base.size(A::AbstractArray2, dim) = Int(ArrayInterface.size(A, dim))

function Base.axes(A::AbstractArray2)
    !(parent_type(A) <: typeof(A)) && return ArrayInterface.axes(parent(A))
    throw(ArgumentError("Subtypes of `AbstractArray2` must define an axes method"))
end
Base.axes(A::AbstractArray2, dim) = ArrayInterface.axes(A, dim)

function Base.strides(A::AbstractArray2)
    defines_strides(A) && return map(Int, ArrayInterface.strides(A))
    throw(MethodError(Base.strides, (A,)))
end
Base.strides(A::AbstractArray2, dim) = Int(ArrayInterface.strides(A, dim))

function Base.IndexStyle(::Type{T}) where {T<:AbstractArray2}
    if parent_type(T) <: T
        return IndexCartesian()
    else
        return IndexStyle(parent_type(T))
    end
end

function Base.length(A::AbstractArray2)
    len = known_length(A)
    if len === missing
        return Int(prod(size(A)))
    else
        return Int(len)
    end
end

@propagate_inbounds Base.getindex(A::AbstractArray2, args...) = getindex(A, args...)
@propagate_inbounds Base.getindex(A::AbstractArray2; kwargs...) = getindex(A; kwargs...)

@propagate_inbounds function Base.setindex!(A::AbstractArray2, val, args...)
    return setindex!(A, val, args...)
end
@propagate_inbounds function Base.setindex!(A::AbstractArray2, val; kwargs...)
    return setindex!(A, val; kwargs...)
end


"""
    is_lazy_conjugate(::AbstractArray) -> Bool

Determine if a given array will lazyily take complex conjugates, such as with `Adjoint`. This will work with
nested wrappers, so long as there is no type in the chain of wrappers such that `parent_type(T) == T`

Examples

    julia> a = transpose([1 + im, 1-im]')
    2×1 transpose(adjoint(::Vector{Complex{Int64}})) with eltype Complex{Int64}:
     1 - 1im
     1 + 1im

    julia> ArrayInterface.is_lazy_conjugate(a)
    True()

    julia> b = a'
    1×2 adjoint(transpose(adjoint(::Vector{Complex{Int64}}))) with eltype Complex{Int64}:
     1+1im  1-1im

    julia> ArrayInterface.is_lazy_conjugate(b)
    False()

"""
is_lazy_conjugate(::T) where {T <: AbstractArray} = _is_lazy_conjugate(T, False())
is_lazy_conjugate(::AbstractArray{<:Real})  = False()

function _is_lazy_conjugate(::Type{T}, isconj) where {T <: AbstractArray}
    Tp = parent_type(T)
    if T !== Tp
        _is_lazy_conjugate(Tp, isconj)
    else
        isconj
    end
end

function _is_lazy_conjugate(::Type{T}, isconj) where {T <: Adjoint}
    Tp = parent_type(T)
    if T !== Tp
        _is_lazy_conjugate(Tp, !isconj)
    else
        !isconj
    end
end

include("ranges.jl")
include("axes.jl")
include("size.jl")
include("dimensions.jl")
include("indexing.jl")
include("stridelayout.jl")
include("broadcast.jl")

function __init__()

    @require SuiteSparse = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9" begin
        function lu_instance(jac_prototype::SparseMatrixCSC)
            return SuiteSparse.UMFPACK.UmfpackLU(
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
    end

    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        ismutable(::Type{<:StaticArrays.StaticArray}) = false
        can_setindex(::Type{<:StaticArrays.StaticArray}) = false
        ismutable(::Type{<:StaticArrays.MArray}) = true
        ismutable(::Type{<:StaticArrays.SizedArray}) = true

        buffer(A::Union{StaticArrays.SArray,StaticArrays.MArray}) = getfield(A, :data)

        function lu_instance(_A::StaticArrays.StaticMatrix{N,N}) where {N}
            A = StaticArrays.SArray(_A)
            L = LowerTriangular(A)
            U = UpperTriangular(A)
            p = StaticArrays.SVector{N,Int}(1:N)
            return StaticArrays.LU(L, U, p)
        end

        function restructure(x::StaticArrays.SArray, y::StaticArrays.SArray)
            return reshape(y, StaticArrays.Size(x))
        end

        function restructure(x::StaticArrays.SArray{S}, y) where {S}
            return StaticArrays.SArray{S}(y)
        end

        known_first(::Type{<:StaticArrays.SOneTo}) = 1
        known_last(::Type{StaticArrays.SOneTo{N}}) where {N} = N
        known_length(::Type{StaticArrays.SOneTo{N}}) where {N} = N
        known_length(::Type{StaticArrays.Length{L}}) where {L} = L
        known_length(::Type{A}) where {A <: StaticArrays.StaticArray} = known_length(StaticArrays.Length(A))

        device(::Type{<:StaticArrays.MArray}) = CPUPointer()
        device(::Type{<:StaticArrays.SArray}) = CPUTuple()
        contiguous_axis(::Type{<:StaticArrays.StaticArray}) = StaticInt{1}()
        contiguous_batch_size(::Type{<:StaticArrays.StaticArray}) = StaticInt{0}()
        function stride_rank(::Type{T}) where {N,T<:StaticArrays.StaticArray{<:Any,<:Any,N}}
            return ArrayInterface.nstatic(Val(N))
        end
        function dense_dims(::Type{<:StaticArrays.StaticArray{S,T,N}}) where {S,T,N}
            return ArrayInterface._all_dense(Val(N))
        end
        defines_strides(::Type{<:StaticArrays.SArray}) = true
        defines_strides(::Type{<:StaticArrays.MArray}) = true

        @generated function axes_types(::Type{<:StaticArrays.StaticArray{S}}) where {S}
            return Tuple{[StaticArrays.SOneTo{s} for s in S.parameters]...}
        end
        @generated function size(A::StaticArrays.StaticArray{S}) where {S}
            t = Expr(:tuple)
            Sp = S.parameters
            for n = 1:length(Sp)
                push!(t.args, Expr(:call, Expr(:curly, :StaticInt, Sp[n])))
            end
            return t
        end
        @generated function strides(A::StaticArrays.StaticArray{S}) where {S}
            t = Expr(:tuple, Expr(:call, Expr(:curly, :StaticInt, 1)))
            Sp = S.parameters
            x = 1
            for n = 1:(length(Sp)-1)
                push!(t.args, Expr(:call, Expr(:curly, :StaticInt, (x *= Sp[n]))))
            end
            return t
        end
        if StaticArrays.SizedArray{Tuple{8,8},Float64,2,2} isa UnionAll
            @inline strides(B::StaticArrays.SizedArray{S,T,M,N,A}) where {S,T,M,N,A<:SubArray} = strides(B.data)
            parent_type(::Type{<:StaticArrays.SizedArray{S,T,M,N,A}}) where {S,T,M,N,A} = A
        else
            parent_type(::Type{<:StaticArrays.SizedArray{S,T,M,N}}) where {S,T,M,N} =
                Array{T,N}
        end
        @require Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e" begin
            function Adapt.adapt_storage(
                ::Type{<:StaticArrays.SArray{S}},
                xs::Array,
            ) where {S}
                return StaticArrays.SArray{S}(xs)
            end
        end
    end

    @require LabelledArrays = "2ee39098-c373-598a-b85f-a56591580800" begin
        ismutable(::Type{<:LabelledArrays.LArray{T,N,Syms}}) where {T,N,Syms} = ismutable(T)
        can_setindex(::Type{<:LabelledArrays.SLArray}) = false
    end

    @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
        ismutable(::Type{<:Tracker.TrackedArray}) = false
        ismutable(T::Type{<:Tracker.TrackedReal}) = false
        can_setindex(::Type{<:Tracker.TrackedArray}) = false
        fast_scalar_indexing(::Type{<:Tracker.TrackedArray}) = false
        aos_to_soa(x::AbstractArray{<:Tracker.TrackedReal,N}) where {N} = Tracker.collect(x)
    end

    @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        @require Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e" begin
            include("cuarrays.jl")
        end
        @require DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e" begin
            # actually do QR
            function lu_instance(A::CuArrays.CuMatrix{T}) where {T}
                return CuArrays.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
            end
        end
    end

    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        @require Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e" begin
            include("cuarrays2.jl")
        end
        @require DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e" begin
            # actually do QR
            function lu_instance(A::CUDA.CuMatrix{T}) where {T}
                return CUDA.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
            end
        end
    end

    @require BandedMatrices = "aae01518-5342-5314-be14-df237901396f" begin
        struct BandedMatrixIndex <: MatrixIndex
            count::Int
            rowsize::Int
            colsize::Int
            bandinds::Array{Int,1}
            bandsizes::Array{Int,1}
            isrow::Bool
        end

        Base.firstindex(i::BandedMatrixIndex) = 1
        Base.lastindex(i::BandedMatrixIndex) = i.count
        Base.length(i::BandedMatrixIndex) = lastindex(i)
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

        function findstructralnz(x::BandedMatrices.BandedMatrix)
            l, u = BandedMatrices.bandwidths(x)
            rowsize, colsize = Base.size(x)
            rowind = BandedMatrixIndex(rowsize, colsize, l, u, true)
            colind = BandedMatrixIndex(rowsize, colsize, l, u, false)
            return (rowind, colind)
        end

        has_sparsestruct(::Type{<:BandedMatrices.BandedMatrix}) = true
        is_structured(::Type{<:BandedMatrices.BandedMatrix}) = true
        fast_matrix_colors(::Type{<:BandedMatrices.BandedMatrix}) = true

        function matrix_colors(A::BandedMatrices.BandedMatrix)
            l, u = BandedMatrices.bandwidths(A)
            width = u + l + 1
            return _cycle(1:width, Base.size(A, 2))
        end

    end

    @require BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0" begin
        @require BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e" begin
            struct BlockBandedMatrixIndex <: MatrixIndex
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

            function findstructralnz(x::BlockBandedMatrices.BlockBandedMatrix)
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
            struct BandedBlockBandedMatrixIndex <: MatrixIndex
                count::Int
                refinds::Array{Int,1}
                refcoords::Array{Int,1}# storing col or row inds at ref points
                reflocalinds::Array{BandedMatrixIndex,1}
                isrow::Bool
            end
            Base.firstindex(i::BandedBlockBandedMatrixIndex) = 1
            Base.lastindex(i::BandedBlockBandedMatrixIndex) = i.count
            Base.length(i::BandedBlockBandedMatrixIndex) = lastindex(i)
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

            function findstructralnz(x::BlockBandedMatrices.BandedBlockBandedMatrix)
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

            has_sparsestruct(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
            has_sparsestruct(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true
            is_structured(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
            is_structured(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true
            fast_matrix_colors(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
            fast_matrix_colors(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true

            function matrix_colors(A::BlockBandedMatrices.BlockBandedMatrix)
                l, u = BlockBandedMatrices.blockbandwidths(A)
                blockwidth = l + u + 1
                nblock = BlockBandedMatrices.blocksize(A, 2)
                cols = BlockArrays.blocklengths(axes(A, 2))
                blockcolors = _cycle(1:blockwidth, nblock)
                # the reserved number of colors of a block is the maximum length of columns of blocks with the same block color
                ncolors = [maximum(cols[i:blockwidth:nblock]) for i = 1:blockwidth]
                endinds = cumsum(ncolors)
                startinds = [endinds[i] - ncolors[i] + 1 for i = 1:blockwidth]
                colors = [
                    (startinds[blockcolors[i]]:endinds[blockcolors[i]])[1:cols[i]]
                    for i = 1:nblock
                ]
                return reduce(vcat,colors)
            end

            function matrix_colors(A::BlockBandedMatrices.BandedBlockBandedMatrix)
                l, u = BlockBandedMatrices.blockbandwidths(A)
                lambda, mu = BlockBandedMatrices.subblockbandwidths(A)
                blockwidth = l + u + 1
                subblockwidth = lambda + mu + 1
                nblock = BlockBandedMatrices.blocksize(A, 2)
                cols = BlockArrays.blocklengths(axes(A, 2))
                blockcolors = _cycle(1:blockwidth, nblock)
                # the reserved number of colors of a block is the min of subblockwidth and the largest length of columns of blocks with the same block color
                ncolors = [
                    min(subblockwidth, maximum(cols[i:blockwidth:nblock]))
                    for i = 1:min(blockwidth, nblock)
                ]
                endinds = cumsum(ncolors)
                startinds = [endinds[i] - ncolors[i] + 1 for i = 1:min(blockwidth, nblock)]
                colors = [
                    _cycle(startinds[blockcolors[i]]:endinds[blockcolors[i]], cols[i])
                    for i = 1:nblock
                ]
                return reduce(vcat,colors)
            end
        end
    end
    @require OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881" begin
        relative_offsets(r::OffsetArrays.IdOffsetRange) = (getfield(r, :offset),)
        relative_offsets(A::OffsetArrays.OffsetArray) = getfield(A, :offsets)
        function relative_offsets(A::OffsetArrays.OffsetArray, ::StaticInt{dim}) where {dim}
            if dim > ndims(A)
                return static(0)
            else
                return getfield(relative_offsets(A), dim)
            end
        end
        function relative_offsets(A::OffsetArrays.OffsetArray, dim::Int)
            if dim > ndims(A)
                return 0
            else
                return getfield(relative_offsets(A), dim)
            end
        end
        ArrayInterface.parent_type(::Type{<:OffsetArrays.OffsetArray{T,N,A}}) where {T,N,A} = A
        function _offset_axis_type(::Type{T}, dim::StaticInt{D}) where {T,D}
            OffsetArrays.IdOffsetRange{Int,ArrayInterface.axes_types(T, dim)}
        end
        function ArrayInterface.axes_types(::Type{T}) where {T<:OffsetArrays.OffsetArray}
            Static.eachop_tuple(_offset_axis_type, Static.nstatic(Val(ndims(T))), ArrayInterface.parent_type(T))
        end
        function ArrayInterface.known_offsets(::Type{A}) where {A<:OffsetArrays.OffsetArray}
            ntuple(identity -> missing, Val(ndims(A)))
        end
        function ArrayInterface.offsets(A::OffsetArrays.OffsetArray)
            map(+, ArrayInterface.offsets(parent(A)), relative_offsets(A))
        end
       @inline function ArrayInterface.offsets(A::OffsetArrays.OffsetArray, dim)
            d = ArrayInterface.to_dims(A, dim)
            ArrayInterface.offsets(parent(A), d) + relative_offsets(A, d)
        end
        @inline function ArrayInterface.axes(A::OffsetArrays.OffsetArray)
            map(OffsetArrays.IdOffsetRange, ArrayInterface.axes(parent(A)), relative_offsets(A))
        end
        @inline function ArrayInterface.axes(A::OffsetArrays.OffsetArray, dim)
            d = to_dims(A, dim)
            OffsetArrays.IdOffsetRange(ArrayInterface.axes(parent(A), d), relative_offsets(A, d))
        end
    end
end

end
