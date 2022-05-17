module ArrayInterfaceCore

using IfElse
using LinearAlgebra
using LinearAlgebra: AbstractTriangular
using SparseArrays
using SuiteSparse
using Static
using Static: Zero, One, nstatic, eq, ne, gt, ge, lt, le, eachop, eachop_tuple,
    find_first_eq, permute, invariant_permutation, field_type, reduce_tup
using Base.Cartesian
import Compat

using Base: @propagate_inbounds, tail, OneTo, LogicalIndex, Slice, ReinterpretArray,
    ReshapedArray, AbstractCartesianIndex

using Base.Iterators: Pairs

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
parent_type(::Type{<:AbstractTriangular{T,S}}) where {T,S} = S
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
can_setindex(::Type{<:Pairs{<:Any,<:Any,P}}) where {P} = can_setindex(P)

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
For `GPUArrays`, will return `ArrayInterfaceCore.GPU()`.
For `AbstractArray` supporting a `pointer` method, returns `ArrayInterfaceCore.CPUPointer()`.
For other `AbstractArray`s and `Tuple`s, returns `ArrayInterfaceCore.CPUIndex()`.
Otherwise, returns `nothing`.
"""
device(A) = device(typeof(A))
device(::Type) = nothing
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

Base.size(A::AbstractArray2) = map(Int, ArrayInterfaceCore.size(A))
Base.size(A::AbstractArray2, dim) = Int(ArrayInterfaceCore.size(A, dim))

function Base.axes(A::AbstractArray2)
    !(parent_type(A) <: typeof(A)) && return ArrayInterfaceCore.axes(parent(A))
    throw(ArgumentError("Subtypes of `AbstractArray2` must define an axes method"))
end
Base.axes(A::AbstractArray2, dim) = ArrayInterfaceCore.axes(A, dim)

function Base.strides(A::AbstractArray2)
    defines_strides(A) && return map(Int, ArrayInterfaceCore.strides(A))
    throw(MethodError(Base.strides, (A,)))
end
Base.strides(A::AbstractArray2, dim) = Int(ArrayInterfaceCore.strides(A, dim))

function Base.IndexStyle(::Type{T}) where {T<:AbstractArray2}
    if parent_type(T) <: T
        return IndexCartesian()
    else
        return IndexStyle(parent_type(T))
    end
end

function Base.length(A::AbstractArray2)
    len = known_length(A)
    if len === nothing
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

    julia> ArrayInterfaceCore.is_lazy_conjugate(a)
    True()

    julia> b = a'
    1×2 adjoint(transpose(adjoint(::Vector{Complex{Int64}}))) with eltype Complex{Int64}:
     1+1im  1-1im

    julia> ArrayInterfaceCore.is_lazy_conjugate(b)
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

end # module
