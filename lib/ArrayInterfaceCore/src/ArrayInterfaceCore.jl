module ArrayInterfaceCore

using LinearAlgebra
using LinearAlgebra: AbstractTriangular
using SparseArrays
using SuiteSparse

@static if isdefined(Base, Symbol("@assume_effects"))
    using Base: @assume_effects
else
    macro assume_effects(_, ex)
        :(Base.@pure $(ex))
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
    ArrayInterfaceCore.map_tuple_type(f, T::Type{<:Tuple})

Returns tuple where each field corresponds to the field type of `T` modified by the function `f`.

# Examples

```julia
julia> ArrayInterfaceCore.map_tuple_type(sqrt, Tuple{1,4,16})
(1.0, 2.0, 4.0)

```
"""
@inline function map_tuple_type(f, @nospecialize(T::Type))
    ntuple(i -> f(fieldtype(T, i)), Val{fieldcount(T)}())
end

"""
    ArrayInterfaceCore.flatten_tuples(t::Tuple) -> Tuple

Flattens any field of `t` that is a tuple. Only direct fields of `t` may be flattened.

# Examples

```julia
julia> ArrayInterfaceCore.flatten_tuples((1, ()))
(1,)

julia> ArrayInterfaceCore.flatten_tuples((1, (2, 3)))
(1, 2, 3)

julia> ArrayInterfaceCore.flatten_tuples((1, (2, (3,))))
(1, 2, (3,))

```
"""
function flatten_tuples(t::Tuple)
    fields = _new_field_positions(t)
    ntuple(Val{nfields(fields)}()) do k
        i, j = getfield(fields, k)
        i = length(t) - i
        @inbounds j === 0 ? getfield(t, i) : getfield(getfield(t, i), j)
    end
end
_new_field_positions(::Tuple{}) = ()
@nospecialize
_new_field_positions(x::Tuple) = (_fl1(x, x[1])..., _new_field_positions(Base.tail(x))...)
_fl1(x::Tuple, x1::Tuple) = ntuple(Base.Fix1(tuple, length(x) - 1), Val(length(x1)))
_fl1(x::Tuple, x1) = ((length(x) - 1, 0),)
@specialize

"""
    parent_type(::Type{T}) -> Type

Returns the parent array that type `T` wraps.
"""
parent_type(x) = parent_type(typeof(x))
parent_type(::Type{Symmetric{T,S}}) where {T,S} = S
parent_type(::Type{<:AbstractTriangular{T,S}}) where {T,S} = S
parent_type(@nospecialize T::Type{<:PermutedDimsArray}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:Adjoint}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:Transpose}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:SubArray}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:Base.ReinterpretArray}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:Base.ReshapedArray}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:Union{Base.Slice,Base.IdentityUnitRange}}) = fieldtype(T, :indices)
parent_type(::Type{Diagonal{T,V}}) where {T,V} = V
parent_type(T::Type) = T

"""
    buffer(x)

Return the buffer data that `x` points to. Unlike `parent(x::AbstractArray)`, `buffer(x)`
may not return another array type.
"""
buffer(x) = parent(x)
buffer(x::SparseMatrixCSC) = getfield(x, :nzval)
buffer(x::SparseVector) = getfield(x, :nzval)
buffer(@nospecialize x::Union{Base.Slice,Base.IdentityUnitRange}) = getfield(x, :indices)

"""
    is_forwarding_wrapper(::Type{T}) -> Bool

Returns `true` if the type `T` wraps another data type and does not alter any of its
standard interface. For example, if `T` were an array then its size, indices, and elements
would all be equivalent to its wrapped data.
"""
is_forwarding_wrapper(T::Type) = false
is_forwarding_wrapper(@nospecialize T::Type{<:Base.Slice}) = true
is_forwarding_wrapper(@nospecialize x) = is_forwarding_wrapper(typeof(x))

"""
    GetIndex(buffer) = GetIndex{true}(buffer)
    GetIndex{check}(buffer) -> g

Wraps an indexable buffer in a function type that is indexed when called, so that `g(inds..)`
is equivalent to `buffer[inds...]`. If `check` is `false`, then all indexing arguments are
considered in-bounds. The default value for `check` is `true`, requiring bounds checking for
each index.

See also [`SetIndex!`](@ref)

!!! Warning
    Passing `false` as `check` may result in incorrect results/crashes/corruption for
    out-of-bounds indices, similar to inappropriate use of `@inbounds`. The user is
    responsible for ensuring this is correctly used.

# Examples

```julia
julia> ArrayInterfaceCore.GetIndex(1:10)(3)
3

julia> ArrayInterfaceCore.GetIndex{false}(1:10)(11)  # shouldn't be in-bounds
11

```

"""
struct GetIndex{CB,B} <: Function
    buffer::B

    GetIndex{true,B}(b) where {B} = new{true,B}(b)
    GetIndex{false,B}(b) where {B} = new{false,B}(b)
    GetIndex{check}(b::B) where {check,B} = GetIndex{check,B}(b)
    GetIndex(b) = GetIndex{true}(b)
end

"""
    SetIndex!(buffer) = SetIndex!{true}(buffer)
    SetIndex!{check}(buffer) -> g

Wraps an indexable buffer in a function type that sets a value at an index when called, so
that `g(val, inds..)` is equivalent to `setindex!(buffer, val, inds...)`. If `check` is
`false`, then all indexing arguments are considered in-bounds. The default value for `check`
is `true`, requiring bounds checking for each index.

See also [`GetIndex`](@ref)

!!! Warning
    Passing `false` as `check` may result in incorrect results/crashes/corruption for
    out-of-bounds indices, similar to inappropriate use of `@inbounds`. The user is
    responsible for ensuring this is correctly used.

# Examples

```julia

julia> x = [1, 2, 3, 4];

julia> ArrayInterface.SetIndex!(x)(10, 2);

julia> x[2]
10

```
"""
struct SetIndex!{CB,B} <: Function
    buffer::B

    SetIndex!{true,B}(b) where {B} = new{true,B}(b)
    SetIndex!{false,B}(b) where {B} = new{false,B}(b)
    SetIndex!{check}(b::B) where {check,B} = SetIndex!{check,B}(b)
    SetIndex!(b) = SetIndex!{true}(b)
end

buffer(x::Union{SetIndex!,GetIndex}) = getfield(x, :buffer)

Base.@propagate_inbounds @inline (g::GetIndex{true})(inds...) = buffer(g)[inds...]
@inline (g::GetIndex{false})(inds...) = @inbounds(buffer(g)[inds...])
Base.@propagate_inbounds @inline function (s::SetIndex!{true})(v, inds...)
    setindex!(buffer(s), v, inds...)
end
@inline (s::SetIndex!{false})(v, inds...) = @inbounds(setindex!(buffer(s), v, inds...))

"""
    can_change_size(::Type{T}) -> Bool

Returns `true` if the Base.size of `T` can change, in which case operations
such as `pop!` and `popfirst!` are available for collections of type `T`.
"""
can_change_size(x) = can_change_size(typeof(x))
function can_change_size(::Type{T}) where {T}
    is_forwarding_wrapper(T) ? can_change_size(parent_type(T)) : false
end
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
can_setindex(T::Type) = is_forwarding_wrapper(T) ? can_setindex(parent_type(T)) : true
can_setindex(@nospecialize T::Type{<:AbstractRange}) = false
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
    device(::Type{T}) -> AbstractDevice

Indicates the most efficient way to access elements from the collection in low-level code.
For `GPUArrays`, will return `ArrayInterface.GPU()`.
For `AbstractArray` supporting a `pointer` method, returns `ArrayInterface.CPUPointer()`.
For other `AbstractArray`s and `Tuple`s, returns `ArrayInterface.CPUIndex()`.
Otherwise, returns `nothing`.
"""
device(A) = device(typeof(A))
device(::Type) = nothing
device(::Type{<:Tuple}) = CPUTuple()
device(::Type{T}) where {T<:Array} = CPUPointer()
device(::Type{T}) where {T<:AbstractArray} = _device(parent_type(T), T)
function _device(::Type{P}, ::Type{T}) where {P,T}
    if defines_strides(T)
        return device(P)
    else
        return _not_pointer(device(P))
    end
end
_not_pointer(::CPUPointer) = CPUIndex()
_not_pointer(x) = x
_device(::Type{T}, ::Type{T}) where {T<:DenseArray} = CPUPointer()
_device(::Type{T}, ::Type{T}) where {T} = CPUIndex()

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

Base.ndims(::ArrayIndex{N}) where {N} = N
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
known_first(T::Type) = is_forwarding_wrapper(T) ? known_first(parent_type(T)) : nothing
known_first(::Type{<:Base.OneTo}) = 1
known_first(@nospecialize T::Type{<:LinearIndices}) = 1
known_first(@nospecialize T::Type{<:Base.IdentityUnitRange}) = known_first(parent_type(T))
function known_first(::Type{<:CartesianIndices{N,R}}) where {N,R}
    _cartesian_index(ntuple(i -> known_first(R.parameters[i]), Val(N)))
end

"""
    known_last(::Type{T}) -> Union{Int,Nothing}

If `last` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

```julia
julia> ArrayInterfaceCore.known_last(typeof(1:4))
nothing

julia> ArrayInterfaceCore.known_first(typeof(static(1):static(4)))
4

```
"""
known_last(x) = known_last(typeof(x))
known_last(T::Type) = is_forwarding_wrapper(T) ? known_last(parent_type(T)) : nothing
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
known_step(T::Type) = is_forwarding_wrapper(T) ? known_step(parent_type(T)) : nothing
known_step(@nospecialize T::Type{<:AbstractUnitRange}) = 1

#=
    stride_preserving_index(::Type{T}) -> StaticBool

Returns `True` if strides between each element can still be derived when indexing with an
instance of type `T`.
=#
stride_preserving_index(@nospecialize T::Type{<:AbstractRange}) = true
stride_preserving_index(@nospecialize T::Type{<:Number}) = true
@inline function stride_preserving_index(@nospecialize T::Type{<:Tuple})
    all(map_tuple_type(stride_preserving_index, T))
end
stride_preserving_index(@nospecialize T::Type) = false

"""
    is_splat_index(::Type{T}) -> Bool

Returns `static(true)` if `T` is a type that splats across multiple dimensions.
"""
is_splat_index(T::Type) = false
is_splat_index(@nospecialize(x)) = is_splat_index(typeof(x))

"""
    ndims_index(::Type{I}) -> Int

Returns the number of dimensions that an instance of `I` indexes into. If this method is
not explicitly defined, then `1` is returned.

See also [`ndims_shape`](@ref)

# Examples

```julia
julia> ArrayInterfaceCore.ndims_index(Int)
1

julia> ArrayInterfaceCore.ndims_index(CartesianIndex(1, 2, 3))
3

julia> ArrayInterfaceCore.ndims_index([CartesianIndex(1, 2), CartesianIndex(1, 3)])
2

```
"""
ndims_index(::Type{<:Base.AbstractCartesianIndex{N}}) where {N} = N
# preserve CartesianIndices{0} as they consume a dimension.
ndims_index(::Type{CartesianIndices{0,Tuple{}}}) = 1
ndims_index(@nospecialize T::Type{<:AbstractArray{Bool}}) = ndims(T)
ndims_index(@nospecialize T::Type{<:AbstractArray}) = ndims_index(eltype(T))
ndims_index(@nospecialize T::Type{<:Base.LogicalIndex}) = ndims(fieldtype(T, :mask))
ndims_index(T::Type) = 1
ndims_index(@nospecialize(i)) = ndims_index(typeof(i))

"""
    ndims_shape(::Type{I}) -> Union{Int,Tuple{Vararg{Int}}}

Returns the number of dimension that are represented in the shape of the returned array when
indexing with an instance of `I`.

See also [`ndims_index`](@ref)

# Examples

```julia
julia> ArrayInterfaceCore.ndims_shape([CartesianIndex(1, 1), CartesianIndex(1, 2)])
1

julia> ndims(CartesianIndices((2,2))[[CartesianIndex(1, 1), CartesianIndex(1, 2)]])
1

"""
ndims_shape(T::DataType) = ndims_index(T)
ndims_shape(::Type{Colon}) = 1
ndims_shape(@nospecialize T::Type{<:CartesianIndices}) = ndims(T)
ndims_shape(@nospecialize T::Type{<:Union{Number,Base.AbstractCartesianIndex}}) = 0
ndims_shape(@nospecialize T::Type{<:AbstractArray{Bool}}) = 1
ndims_shape(@nospecialize T::Type{<:AbstractArray}) = ndims(T)
ndims_shape(x) = ndims_shape(typeof(x))

@assume_effects :total function _find_first_true(isi::Tuple{Vararg{Bool,N}}) where {N}
    for i in 1:N
        getfield(isi, i) && return i
    end
    return nothing
end

"""
    IndicesInfo{N}(T::Type{<:Tuple}) -> IndicesInfo{N,pdims,cdims}()

Provides basic trait information for each index type in in the tuple `T`. `pdims` and
`cdims` are dimension mappings to the parent and child dimensions respectively.

# Examples

```julia
julia> using ArrayInterfaceCore: IndicesInfo

julia> IndicesInfo{5}(typeof((:,[CartesianIndex(1,1),CartesianIndex(1,1)], 1, ones(Int, 2, 2), :, 1)))
IndicesInfo{5, (1, (2, 3), 4, 5, 0, 0), (1, 2, 0, (3, 4), 5, 0)}()

```
"""
struct IndicesInfo{N,NI,NS} end
IndicesInfo(x::SubArray) = IndicesInfo{ndims(parent(x))}(typeof(x.indices))
@inline function IndicesInfo(@nospecialize T::Type{<:SubArray})
    IndicesInfo{ndims(parent_type(T))}(fieldtype(T, :indices))
end
function IndicesInfo{N}(@nospecialize(T::Type{<:Tuple})) where {N}
    _indices_info(
        Val{_find_first_true(map_tuple_type(is_splat_index, T))}(),
        IndicesInfo{N,map_tuple_type(ndims_index, T),map_tuple_type(ndims_shape, T)}()
    )
end
function _indices_info(::Val{nothing}, ::IndicesInfo{1,(1,),NS}) where {NS}
    ns1 = getfield(NS, 1)
    IndicesInfo{1,(1,), (ns1 > 1 ? ntuple(identity, ns1) : ns1,)}()
end
function _indices_info(::Val{nothing}, ::IndicesInfo{N,(1,),NS}) where {N,NS}
    ns1 = getfield(NS, 1)
    IndicesInfo{N,(:,),(ns1 > 1 ? ntuple(identity, ns1) : ns1,)}()
end
@inline function _indices_info(::Val{nothing}, ::IndicesInfo{N,NI,NS}) where {N,NI,NS}
    if sum(NI) > N
        IndicesInfo{N,_replace_trailing(N, _accum_dims(cumsum(NI), NI)), _accum_dims(cumsum(NS), NS)}()
    else
        IndicesInfo{N,_accum_dims(cumsum(NI), NI), _accum_dims(cumsum(NS), NS)}()
    end
end
@inline function _indices_info(::Val{SI}, ::IndicesInfo{N,NI,NS}) where {N,NI,NS,SI}
    nsplat = N - sum(NI)
    if nsplat === 0
        _indices_info(Val{nothing}(), IndicesInfo{N,NI,NS}())
    else
        splatmul = max(0, nsplat + 1)
        _indices_info(Val{nothing}(), IndicesInfo{N,_map_splats(splatmul, SI, NI),_map_splats(splatmul, SI, NS)}())
    end
end
@inline function _map_splats(nsplat::Int, splat_index::Int, dims::Tuple{Vararg{Int}})
    ntuple(length(dims)) do i
        i === splat_index ? (nsplat * getfield(dims, i)) : getfield(dims, i)
    end
end
@inline function _replace_trailing(n::Int, dims::Tuple{Vararg{Any,N}}) where {N}
    ntuple(N) do i
        dim_i = getfield(dims, i)
        if dim_i isa Tuple
            ntuple(length(dim_i)) do j
                dim_i_j = getfield(dim_i, j)
                dim_i_j > n ? 0 : dim_i_j
            end
        else
            dim_i > n ? 0 : dim_i
        end
    end
end
@inline function _accum_dims(csdims::NTuple{N,Int}, nd::NTuple{N,Int}) where {N}
    ntuple(N) do i
        nd_i = getfield(nd, i)
        if nd_i === 0
            0
        elseif nd_i === 1
            getfield(csdims, i)
        else
            ntuple(Base.Fix1(+, getfield(csdims, i) - nd_i), nd_i)
        end
    end
end

"""
    instances_do_not_alias(::Type{T}) -> Bool

Is it safe to `ivdep` arrays containing elements of type `T`?
That is, would it be safe to write to an array full of `T` in parallel?
This is not true for `mutable struct`s in general, where editing one index
could edit other indices.
That is, it is not safe when different instances may alias the same memory.
"""
instances_do_not_alias(::Type{T}) where {T} = Base.isbitstype(T)

"""
    indices_do_not_alias(::Type{T<:AbstractArray}) -> Bool

Is it safe to `ivdep` arrays of type `T`?
That is, would it be safe to write to an array of type `T` in parallel?
Examples where this is not true are `BitArray`s or `view(rand(6), [1,2,3,1,2,3])`.
That is, it is not safe whenever different indices may alias the same memory.
"""
indices_do_not_alias(::Type) = false
indices_do_not_alias(::Type{A}) where {T, A<:Base.StridedArray{T}} = instances_do_not_alias(T)
indices_do_not_alias(::Type{Adjoint{T,A}}) where {T, A <: AbstractArray{T}} = indices_do_not_alias(A)
indices_do_not_alias(::Type{Transpose{T,A}}) where {T, A <: AbstractArray{T}} = indices_do_not_alias(A)
indices_do_not_alias(::Type{<:SubArray{<:Any,<:Any,A,I}}) where {
  A,I<:Tuple{Vararg{Union{Integer, UnitRange, Base.ReshapedUnitRange, Base.AbstractCartesianIndex}}}} = indices_do_not_alias(A)

"""
    known_dimnames(::Type{T}) -> Tuple{Vararg{Union{Symbol,Nothing}}}
    known_dimnames(::Type{T}, dim::Union{Int,StaticInt}) -> Union{Symbol,Nothing}

Return the names of the dimensions for `x`. `:_` is used to indicate a dimension does not
have a name.
"""
@inline known_dimnames(x, dim::Int) = _itrndims(x) < dim ? :_ : getfield(known_dimnames(x), dim)
known_dimnames(x) = known_dimnames(typeof(x))
function known_dimnames(@nospecialize T::Type{<:VecAdjTrans})
    (:_, getfield(known_dimnames(parent_type(T)), 1))
end
function known_dimnames(@nospecialize T::Type{<:MatAdjTrans})
    n1, n2 = known_dimnames(T)
    (n2, n1)
end
_permdims(::Type{<:PermutedDimsArray{<:Any,<:Any,I1,I2}}) where {I1,I2} = (I1, I2)
function known_dimnames(@nospecialize T::Type{<:PermutedDimsArray})
    map(GetIndex{false}(known_dimnames(parent_type(T))), getfield(_permdims(T), 1))
end
known_dimnames(@nospecialize T::Type{<:SubArray}) = _sub_known_dimnames(IndicesInfo(T), T)
function _sub_known_dimnames(::IndicesInfo{N,pdims,cdims}, @nospecialize(T::Type{<:SubArray})) where {N,pdims,cdims}
    indices_dimnames = map_tuple_type(_known_index_dimnames, fieldtype(T, :indices))
    parent_dimnames = known_dimnames(parent_type(T))
    flatten_tuples(ntuple(Val{nfields(pdims)}()) do index
        _sub_dimname(parent_dimnames, getfield(indices_dimnames, index), getfield(pdims, index))
    end)
end
_known_index_dimnames(@nospecialize T::Type) = _known_index_dimnames(known_dimnames(T))
_known_index_dimnames(@nospecialize dnames::Tuple{Symbol,Vararg{Symbol}}) = dnames
_known_index_dimnames(dnames::Tuple{Symbol}) = first(dnames)
_known_index_dimnames(::Tuple{}) = ()
_sub_dimname(p::Tuple{Vararg{Symbol}}, n::Symbol, d::Int) = n === :_ ? getfield(p, d) : n
_sub_dimname(@nospecialize(p::Tuple{Vararg{Symbol}}), n::Symbol, @nospecialize(d::Tuple{Vararg{Int}})) = n
_sub_dimname(@nospecialize(p::Tuple{Vararg{Symbol}}), n::Tuple{Vararg{Symbol}}, d::Int) = n
_sub_dimname(@nospecialize(p::Tuple{Vararg{Symbol}}), n::Tuple{Vararg{Symbol}}, @nospecialize(d::Tuple{Vararg{Int}})) = n
function known_dimnames(@nospecialize T::Type{<:Base.NonReshapedReinterpretArray})
    known_dimnames(parent_type(T))
end
function known_dimnames(@nospecialize T::Type{<:Base.ReshapedReinterpretArray})
    ss = sizeof(eltype(parent_type(T)))
    ts = sizeof(eltype(T))
    if ss === ts
        return known_dimnames(parent_type(T))
    elseif ss > ts
        return (:_, known_dimnames(parent_type(T))...)
    else
        return Base.tail(known_dimnames(parent_type(T)))
    end
end
@inline function known_dimnames(@nospecialize T::Type{<:Base.ReshapedArray})
    if ndims(T) === ndims(parent_type(T))
        return known_dimnames(parent_type(T))
    elseif ndims(T) > ndims(parent_type(T))
        return (known_dimnames(parent_type(T))..., ntuple(_->:_, Val{ndims(T) - ndims(parent_type(T))}())...)
    else
        return ntuple(_->:_, Val{ndims(T)}())
    end
end
@inline function known_dimnames(::Type{T}) where {T}
    if is_forwarding_wrapper(T)
        return known_dimnames(parent_type(T))
    else
        ntuple(_->:_, Val{_itrndims(T)}())
    end
end

_itrndims(@nospecialize x) = Base.IteratorSize(x) isa Base.HasShape ? ndims(x) : 1

"""
    known_offsets(::Type{T}) -> Tuple
    known_offsets(::Type{T}, dim) -> Union{Int,Nothing}

Returns a tuple of offset values known at compile time. If the offset of a given axis is
not known at compile time `nothing` is returned its position.
"""
known_offsets(x, dim::Int) = ndims(x) < dim ? 1 : getfield(known_offsets(x), dim)
known_offsets(x, s::Symbol) = known_offsets(x, Base.sym_in(s, known_dimnames(x)))
known_offsets(x) = known_offsets(typeof(x))
function known_offsets(T::Type)
    if is_forwarding_wrapper(T)
        known_offsets(parent_type(T))
    else
        ntuple(_->1, Val{_itrndims(T)}())
    end
end
known_offsets(@nospecialize T::Type{<:Number}) = ()  # Int has no dimensions
@inline function known_offsets(@nospecialize T::Type{<:SubArray})
    flatten_tuples(map_tuple_type(known_offsets, fieldtype(T, :indices)))
end
known_offsets(@nospecialize T::Type{<:VecAdjTrans}) = (1, known_offset1(parent_type(T)))
function known_offsets(@nospecialize T::Type{<:MatAdjTrans})
    o1, o2 = known_offsets(parent_type(T))
    (o2, o1)
end
@inline function known_offsets(@nospecialize T::Type{<:PermutedDimsArray})
    map(GetIndex{false}(known_offsets(parent_type(T))), getfield(_permdims(T), 1))
end
function known_offsets(@nospecialize T::Type{<:Base.ReshapedReinterpretArray})
    tcs = sizeof(eltype(T))  # child eltype size
    tps = sizeof(eltype(parent_type(T)))  # parent eltype size
    if tps > tcs
        return (1, known_offsets(parent_type(T))...)
    elseif tcs === tps
        return known_offsets(parent_type(T))
    else
        return Base.tail(known_offsets(parent_type(T)))
    end

end
function known_offsets(@nospecialize T::Type{<:Base.NonReshapedReinterpretArray})
    tcs = sizeof(eltype(T))  # child eltype size
    tps = sizeof(eltype(parent_type(T)))  # parent eltype size
    if tcs === tps
        return known_offsets(parent_type(T))
    else
        return (1, Base.tail(known_offsets(parent_type(T)))...)
    end
end

"""
    known_offset1(::Type{T}) -> Union{Int,Nothing}

Returns the linear offset of array `x` if known at compile time.
"""
@inline known_offset1(x) = known_offsets(x, 1)

"""
    known_size(::Type{T}) -> Tuple
    known_size(::Type{T}, dim) -> Union{Int,Nothing}

Returns the size of each dimension of `A` or along dimension `dim` of `A` that is known at
compile time. If a dimension does not have a known size along a dimension then `nothing` is
returned in its position.
"""
known_size(x) = known_size(typeof(x))
@inline function known_size(::Type{T}) where {T}
    if is_forwarding_wrapper(T)
        return known_size(parent_type(T))
    elseif Base.IteratorSize(T) isa Base.HasShape
        return ntuple(_->nothing, ndims(T))
    else
        return (known_length(T),)
    end
end
known_size(x, d::Int) = ndims(x) < d ? 1 : getfield(known_size(x), d)
known_size(x, d::Symbol) = known_size(x, Base.sym_in(d, known_dimnames(d)))
@inline known_size(@nospecialize T::Type{<:Number}) = ()
@inline known_size(@nospecialize T::Type{<:VecAdjTrans}) = (1, known_length(parent_type(T)))
@inline function known_size(@nospecialize T::Type{<:MatAdjTrans})
    s1, s2 = known_size(parent_type(T))
    (s2, s1)
end
@inline function known_size(@nospecialize T::Type{<:PermutedDimsArray})
    map(GetIndex{false}(known_size(parent_type(T))), getfield(_permdims(T), 1))
end
function known_size(@nospecialize T::Type{<:Diagonal})
    s = known_length(parent_type(T))
    (s, s)
end
known_size(@nospecialize T::Type{<:Union{Symmetric,Hermitian}}) = known_size(parent_type(T))
@inline function known_size(@nospecialize T::Type{<:Base.ReshapedReinterpretArray})
    tcs = sizeof(eltype(T))  # child eltype size
    tps = sizeof(eltype(parent_type(T)))  # parent eltype size
    if tps > tcs
        return flatten_tuples((div(tps, tcs), known_size(parent_type(T))))
    elseif tcs === tps
        return known_size(parent_type(T))
    else
        return Base.tail(known_size(parent_type(T)))
    end
end
@inline function known_size(@nospecialize T::Type{<:Base.NonReshapedReinterpretArray})
    psize = known_size(parent_type(T))
    if Base.issingletontype(eltype(T)) || first(psize) === nothing
        return psize
    else
        return flatten_tuples((div(first(psize) * sizeof(eltype(parent_type(T))), sizeof(eltype(T))), Base.tail(psize)))
    end
end
known_size(@nospecialize T::Type{<:Base.IdentityUnitRange}) = known_size(parent_type(T))
known_size(::Type{<:Base.Generator{I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Reverse{I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Enumerate{I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Accumulate{<:Any,I}}) where {I} = known_size(I)
known_size(::Type{<:Iterators.Pairs{<:Any,<:Any,I}}) where {I} = known_size(I)
@inline function known_size(::Type{<:Iterators.ProductIterator{T}}) where {T}
    ntuple(i -> known_length(T.parameters[i]), Val(known_length(T)))
end
@inline function known_size(@nospecialize T::Type{<:AbstractRange})
    start = known_first(T)
    s = known_step(T)
    stop = known_last(T)
    if start === nothing || s === nothing || stop === nothing
        return is_forwarding_wrapper(T) ? known_size(parent_type(T)) : (nothing,)
    elseif s > 0
        return (max(0, div(stop - start, s) + 1),)
    else
        return (max(0, div(start - stop, -s) + 1),)
    end
end
@inline function known_size(@nospecialize T::Type{<:Union{LinearIndices,CartesianIndices}})
    map_tuple_type(known_length, fieldtype(T, :indices))
end
known_size(@nospecialize T::Type{<:SubArray}) = _sub_known_size(IndicesInfo(T), T)
function _sub_known_size(::IndicesInfo{N,pdims}, @nospecialize(T::Type{<:SubArray})) where {N,pdims}
    P = parent_type(T)
    I = fieldtype(T, :indices)
    flatten_tuples(ntuple(Val{nfields(pdims)}()) do i
        I_i = fieldtype(I, i)
        if I_i <: Base.Slice{Base.OneTo{Int}}
            known_size(P, getfield(pdims, i))
        else
            known_size(I_i)
        end
    end)
end

# 1. `Zip` doesn't check that its collections are compatible (same size) at construction,
#   but we assume as much b/c otherwise it will error while iterating. So we promote to the
#   known size if matching a `Nothing` and `Int` size.
# 2. `promote_shape(::Tuple{Vararg{CanonicalInt}}, ::Tuple{Vararg{CanonicalInt}})` promotes
#   trailing dimensions (which must be of size 1), to `static(1)`. We want to stick to
#   `Nothing` and `Int` types, so we do one last pass to ensure everything is dynamic
@inline function known_size(::Type{<:Iterators.Zip{T}}) where {T}
    N = known_length(T)
    if N > 0
        szs = map_tuple_type(known_size, T)
        return _combine_sizes(first(szs), Base.tail(szs))
    else
        return ()
    end
end
_combine_size(::Nothing, ::Nothing) = nothing
_combine_size(x::Int, ::Nothing) = x
_combine_size(::Nothing, y::Int) = y
_combine_size(::Int, y::Int) = y
@inline function _combine_size(x::Tuple{Vararg{Any,Nx}}, y::Tuple{Vararg{Any,Ny}}) where {Nx,Ny}
    if Nx >= Ny
        ntuple(Val{Nx}()) do i
            _combine_size(getfield(x, i), i > Ny ? 1 : getfield(y, i))
        end
    else
        return _combine_size(y, x)
    end
end
_combine_sizes(sz::Tuple, ::Tuple{}) = sz
function _combine_sizes(sz::Tuple, szs::Tuple)
    _combine_sizes(_combine_size(sz, first(szs)), Base.tail(szs))
end

"""
    known_length(::Type{T}) -> Union{Int,Nothing}

If `length` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.
"""
known_length(x) = known_length(typeof(x))
known_length(@nospecialize T::Type{<:Number}) = 1
known_length(@nospecialize T::Type{<:Union{NamedTuple,Tuple}}) = fieldcount(T)
known_length(@nospecialize T::Type{<:Base.Slice}) = known_length(parent_type(T))
known_length(::Type{<:Base.AbstractCartesianIndex{N}}) where {N} = N
function known_length(::Type{T}) where {T}
    if Base.IteratorSize(T) isa Base.HasShape
        _prod_or_nothing(known_size(T))
    else
        return nothing
    end
end
function known_length(::Type{<:Iterators.Flatten{I}}) where {I}
  _prod_or_nothing((known_length(I), known_length(eltype(I))))
end
_prod_or_nothing(x::Tuple{Vararg{Int}}) = prod(x)
_prod_or_nothing(_) = nothing

"""
    known_strides(::Type{T}) -> Tuple
    known_strides(::Type{T}, dim) -> Union{Int,Nothing}

Returns the strides of array `A` known at compile time. Any strides that are not known at
compile time are represented by `nothing`.
"""
known_strides(x) = known_strides(typeof(x))
function known_strides(T::Type)
    if is_forwarding_wrapper(T)
        return known_strides(parent_type(T))
    elseif defines_strides(T)
        return size_to_strides(known_size(T), 1)
    else
        return ntuple(_->:_, _itrndims(T))
    end
end
# see https://github.com/JuliaLang/julia/blob/6468dcb04ea2947f43a11f556da9a5588de512a0/base/reinterpretarray.jl#L148
# for original code in Base that gives strides by individual dimensions
known_strides(x, d::Int) = ndims(x) < d ? known_length(x) : getfield(known_strides(x), d)
known_strides(x, s::Symbol) = known_strides(x, Base.sym_in(s, known_dimnames(x)))
known_strides(::Type{T}) where {T<:Vector} = (1,)
@inline function known_strides(@nospecialize T::Type{<:VecAdjTrans})
    strd = first(known_strides(parent_type(T)))
    return (strd, strd)
end
@inline function known_strides(@nospecialize T::Type{<:MatAdjTrans})
    s1, s2 = known_strides(parent_type(T))
    (s2, s1)
end
@inline function known_strides(@nospecialize T::Type{<:PermutedDimsArray})
    map(GetIndex{false}(known_strides(parent_type(T))), getfield(_permdims(T), 1))
end
# FIXME
@inline function known_strides(@nospecialize T::Type{<:SubArray})
    if defines_strides(T)
        _sub_known_strides(IndicesInfo(T), T)
    else
        ArgumentError("Provided type does not support strides.") |> throw
    end
end
function _sub_known_strides(::IndicesInfo{N,pdims,cdims}, @nospecialize(T::Type{<:SubArray})) where {N,pdims,cdims}
    steps = map_tuple_type(_try_known_step, fieldtype(T, :indices))
    strs = known_strides(parent_type(T))
    ntuple(Val{nfields(pdims)}()) do i
        if getfield(cdims, 1) === 0
            ()
        else
            _mul(getfield(steps, i), getfield(strs, getfield(pdims, i)))
        end
    end
end
_try_known_step(@nospecialize x) = known_step(x)


function size_to_strides(sz::S, init) where {N,S<:Tuple{Vararg{Any,N}}}
    if @generated
        out = Expr(:block, Expr(:meta, :inline))
        t = Expr(:tuple, :init)
        prev = :init
        i = 1
        while i <= (N - 1)
            if S.parameters[i] <: Nothing || (i > 1 &&  t.args[i - 1] === :nothing)
                push!(t.args, :nothing)
            else
                next = Symbol(:val_, i)
                push!(out.args, :($next = $prev * getfield(sz, $i)))
                push!(t.args, next)
                prev = next
            end
            i += 1
        end
        push!(out.args, t)
        return out
    else
        return _size_to_strides(init, sz...)
    end
end

_mul(x, y) = x * y
_mul(::Nothing, ::Nothing) = nothing
_mul(x, ::Nothing) = nothing
_mul(::Nothing, y) = nothing

@inline _size_to_strides(s, d, sz...) = (s, _size_to_strides(_mul(s, d), sz...)...)
_size_to_strides(s, d) = (s,)
_size_to_strides(s) = ()

"""
    defines_strides(::Type{T}) -> Bool

Is strides(::T) defined? It is assumed that types returning `true` also return a valid
pointer on `pointer(::T)`.
"""
defines_strides(x) = defines_strides(typeof(x))
_defines_strides(::Type{T}, ::Type{T}) where {T} = false
_defines_strides(::Type{P}, ::Type{T}) where {P,T} = defines_strides(P)
defines_strides(::Type{T}) where {T} = _defines_strides(parent_type(T), T)
defines_strides(@nospecialize T::Type{<:StridedArray}) = true
defines_strides(@nospecialize T::Type{<:BitArray}) = true
@inline function defines_strides(@nospecialize T::Type{<:SubArray})
    stride_preserving_index(fieldtype(T, :indices))
end

end # module
