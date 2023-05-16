module ArrayInterface

using LinearAlgebra
using SparseArrays
using SuiteSparse

@static if isdefined(Base, Symbol("@assume_effects"))
    using Base: @assume_effects
else
    macro assume_effects(args...)
        n = nfields(args)
        call = getfield(args, n)
        if n === 2 && getfield(args, 1) === QuoteNode(:total)
            return esc(:(Base.@pure $(call)))
        else
            return esc(call)
        end
    end
end
@assume_effects :total __parameterless_type(T)=Base.typename(T).wrapper
parameterless_type(x) = parameterless_type(typeof(x))
parameterless_type(x::Type) = __parameterless_type(x)

const VecAdjTrans{T, V <: AbstractVector{T}} = Union{Transpose{<:Any, V}, Adjoint{<:Any, V}}
const MatAdjTrans{T, M <: AbstractMatrix{T}} = Union{Transpose{<:Any, M}, Adjoint{<:Any, M}}
const UpTri{T, M} = Union{UpperTriangular{T, M}, UnitUpperTriangular{T, M}}
const LoTri{T, M} = Union{LowerTriangular{T, M}, UnitLowerTriangular{T, M}}

"""
    ArrayInterface.map_tuple_type(f, T::Type{<:Tuple})

Returns tuple where each field corresponds to the field type of `T` modified by the function `f`.

# Examples

```julia
julia> ArrayInterface.map_tuple_type(sqrt, Tuple{1,4,16})
(1.0, 2.0, 4.0)

```
"""
function map_tuple_type end
if VERSION >= v"1.8"
    @inline function map_tuple_type(f, @nospecialize(T::Type))
        ntuple(i -> f(fieldtype(T, i)), Val{fieldcount(T)}())
    end
else
    function map_tuple_type(f::F, ::Type{T}) where {F, T <: Tuple}
        if @generated
            t = Expr(:tuple)
            for i in 1:fieldcount(T)
                push!(t.args, :(f($(fieldtype(T, i)))))
            end
            Expr(:block, Expr(:meta, :inline), t)
        else
            Tuple(f(fieldtype(T, i)) for i in 1:fieldcount(T))
        end
    end
end

"""
    ArrayInterface.flatten_tuples(t::Tuple) -> Tuple

Flattens any field of `t` that is a tuple. Only direct fields of `t` may be flattened.

# Examples

```julia
julia> ArrayInterface.flatten_tuples((1, ()))
(1,)

julia> ArrayInterface.flatten_tuples((1, (2, 3)))
(1, 2, 3)

julia> ArrayInterface.flatten_tuples((1, (2, (3,))))
(1, 2, (3,))

```
"""
function flatten_tuples end
if VERSION >= v"1.8"
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
    function _new_field_positions(x::Tuple)
        (_fl1(x, x[1])..., _new_field_positions(Base.tail(x))...)
    end
    _fl1(x::Tuple, x1::Tuple) = ntuple(Base.Fix1(tuple, length(x) - 1), Val(length(x1)))
    _fl1(x::Tuple, x1) = ((length(x) - 1, 0),)
    @specialize
else
    @inline function flatten_tuples(t::Tuple)
        if @generated
            texpr = Expr(:tuple)
            for i in 1:fieldcount(t)
                p = fieldtype(t, i)
                if p <: Tuple
                    for j in 1:fieldcount(p)
                        push!(texpr.args, :(@inbounds(getfield(getfield(t, $i), $j))))
                    end
                else
                    push!(texpr.args, :(@inbounds(getfield(t, $i))))
                end
            end
            Expr(:block, Expr(:meta, :inline), texpr)
        else
            _flatten(t)
        end
    end
    _flatten(::Tuple{}) = ()
    @inline function _flatten(t::Tuple{Any, Vararg{Any}})
        (getfield(t, 1), _flatten(Base.tail(t))...)
    end
    @inline function _flatten(t::Tuple{Tuple, Vararg{Any}})
        (getfield(t, 1)..., _flatten(Base.tail(t))...)
    end
end

"""
    parent_type(::Type{T}) -> Type

Returns the parent array that type `T` wraps.
"""
parent_type(x) = parent_type(typeof(x))
parent_type(@nospecialize T::Type{<:Union{Symmetric, Hermitian}}) = fieldtype(T, :data)
parent_type(@nospecialize T::Type{<:Union{UpTri, LoTri}}) = fieldtype(T, :data)
parent_type(@nospecialize T::Type{<:PermutedDimsArray}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:Adjoint}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:Transpose}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:SubArray}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:Base.ReinterpretArray}) = fieldtype(T, :parent)
parent_type(@nospecialize T::Type{<:Base.ReshapedArray}) = fieldtype(T, :parent)
function parent_type(@nospecialize T::Type{<:Union{Base.Slice, Base.IdentityUnitRange}})
    fieldtype(T, :indices)
end
parent_type(@nospecialize T::Type{<:Diagonal}) = fieldtype(T, :diag)
parent_type(T::Type) = T

"""
    promote_eltype(::Type{<:AbstractArray{T,N}}, ::Type{T2})

Computes the type of the `AbstractArray` that results from the element
type changing to `promote_type(T,T2)`.

Note that no generic fallback is given.
"""
function promote_eltype end
function promote_eltype(::Type{Array{T, N}}, ::Type{T2}) where {T, T2, N}
    Array{promote_type(T, T2), N}
end

"""
    buffer(x)

Return the buffer data that `x` points to. Unlike `parent(x::AbstractArray)`, `buffer(x)`
may not return another array type.
"""
buffer(x) = parent(x)
buffer(x::SparseMatrixCSC) = getfield(x, :nzval)
buffer(x::SparseVector) = getfield(x, :nzval)
buffer(@nospecialize x::Union{Base.Slice, Base.IdentityUnitRange}) = getfield(x, :indices)

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

!!! warning
    Passing `false` as `check` may result in incorrect results/crashes/corruption for
    out-of-bounds indices, similar to inappropriate use of `@inbounds`. The user is
    responsible for ensuring this is correctly used.

# Examples

```julia
julia> ArrayInterface.GetIndex(1:10)(3)
3

julia> ArrayInterface.GetIndex{false}(1:10)(11)  # shouldn't be in-bounds
11

```

"""
struct GetIndex{CB, B} <: Function
    buffer::B

    GetIndex{true, B}(b) where {B} = new{true, B}(b)
    GetIndex{false, B}(b) where {B} = new{false, B}(b)
    GetIndex{check}(b::B) where {check, B} = GetIndex{check, B}(b)
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

!!! warning
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
struct SetIndex!{CB, B} <: Function
    buffer::B

    SetIndex!{true, B}(b) where {B} = new{true, B}(b)
    SetIndex!{false, B}(b) where {B} = new{false, B}(b)
    SetIndex!{check}(b::B) where {check, B} = SetIndex!{check, B}(b)
    SetIndex!(b) = SetIndex!{true}(b)
end

buffer(x::Union{SetIndex!, GetIndex}) = getfield(x, :buffer)

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
function ismutable(::Type{T}) where {T <: AbstractArray}
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
can_setindex(::Type{<:Base.Iterators.Pairs{<:Any, <:Any, P}}) where {P} = can_setindex(P)

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
issingular(A::Union{Hermitian, Symmetric}) = diaganyzero(bunchkaufman(A, check = false).LD)
issingular(A::Union{LowerTriangular, UpperTriangular}) = diaganyzero(A.data)
issingular(A::Union{UnitLowerTriangular, UnitUpperTriangular}) = false
issingular(A::Union{Adjoint, Transpose}) = issingular(parent(A))
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

function findstructralnz(x::Union{Tridiagonal, SymTridiagonal})
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
function fast_matrix_colors(A::Type{
                                    <:Union{Diagonal, Bidiagonal, Tridiagonal,
                                            SymTridiagonal}})
    true
end

"""
    matrix_colors(A::Union{Array,UpperTriangular,LowerTriangular})

The color vector for dense matrix and triangular matrix is simply
`[1,2,3,..., Base.size(A,2)]`.
"""
function matrix_colors(A::Union{Array, UpperTriangular, LowerTriangular})
    eachindex(1:Base.size(A, 2)) # Vector Base.size matches number of rows
end
matrix_colors(A::Diagonal) = fill(1, Base.size(A, 2))
matrix_colors(A::Bidiagonal) = _cycle(1:2, Base.size(A, 2))
matrix_colors(A::Union{Tridiagonal, SymTridiagonal}) = _cycle(1:3, Base.size(A, 2))
_cycle(repetend, len) = repeat(repetend, div(len, length(repetend)) + 1)[1:len]

"""
bunchkaufman_instance(A, pivot = LinearAlgebra.RowMaximum()) -> bunchkaufman_factorization_instance

Returns an instance of the Bunch-Kaufman factorization object with the correct type
cheaply.
"""
function bunchkaufman_instance(A::Matrix{T}) where T
    return bunchkaufman(similar(A, 0, 0), check = false)
end
function bunchkaufman_instance(A::SparseMatrixCSC)
    bunchkaufman(sparse(similar(A, 1, 1)), check = false)
end

"""
bunchkaufman_instance(a::Number) -> a

Returns the number.
"""
bunchkaufman_instance(a::Number) = a

"""
bunchkaufman_instance(a::Any) -> cholesky(a, check=false)

Returns the number.
"""
bunchkaufman_instance(a::Any) = bunchkaufman(a, check = false)

"""
cholesky_instance(A, pivot = LinearAlgebra.RowMaximum()) -> cholesky_factorization_instance

Returns an instance of the Cholesky factorization object with the correct type
cheaply.
"""
function cholesky_instance(A::Matrix{T}, pivot = LinearAlgebra.RowMaximum()) where {T}  
    return cholesky(similar(A, 0, 0), pivot, check = false)
end
function cholesky_instance(A::SparseMatrixCSC, pivot = LinearAlgebra.RowMaximum())
    cholesky(sparse(similar(A, 1, 1)), check = false)
end

"""
cholesky_instance(a::Number, pivot = LinearAlgebra.RowMaximum()) -> a

Returns the number.
"""
cholesky_instance(a::Number, pivot = LinearAlgebra.RowMaximum()) = a

"""
cholesky_instance(a::Any, pivot = LinearAlgebra.RowMaximum()) -> cholesky(a, check=false)

Slow fallback which gets the instance via factorization. Should get
specialized for new matrix types.
"""
cholesky_instance(a::Any, pivot = LinearAlgebra.RowMaximum()) = cholesky(a, pivot, check = false)

"""
ldlt_instance(A) -> ldlt_factorization_instance

Returns an instance of the LDLT factorization object with the correct type
cheaply.
"""
function ldlt_instance(A::Matrix{T}) where {T}  
    return ldlt(SymTridiagonal(similar(A, 0, 0)), check = false)
end
function ldlt_instance(A::SparseMatrixCSC)
    ldlt(sparse(similar(A, 1, 1)), check = false)
end

"""
ldlt_instance(a::Number) -> a

Returns the number.
"""
ldlt_instance(a::Number) = a

"""
ldlt_instance(a::Any) -> ldlt(a, check=false)

Slow fallback which gets the instance via factorization. Should get
specialized for new matrix types.
"""
ldlt_instance(a::Any) = ldlt(a)

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
    @static if VERSION < v"1.9.0-DEV.1622"
        SuiteSparse.UMFPACK.UmfpackLU(Ptr{Cvoid}(),
                                      Ptr{Cvoid}(),
                                      1,
                                      1,
                                      jac_prototype.colptr[1:1],
                                      jac_prototype.rowval[1:1],
                                      jac_prototype.nzval[1:1],
                                      0)
    else
        SuiteSparse.UMFPACK.UmfpackLU(similar(jac_prototype, 1, 1))
    end
end

"""
  lu_instance(a::Number) -> a

Returns the number.
"""
lu_instance(a::Number) = a

"""
    lu_instance(a::Any) -> lu(a, check=false)

Slow fallback which gets the instance via factorization. Should get
specialized for new matrix types.
"""
lu_instance(a::Any) = lu(a, check = false)

"""
  qr_instance(A) -> qr_factorization_instance

Returns an instance of the QR factorization object with the correct type
cheaply.
"""
function qr_instance(A::Matrix{T}) where {T}
    LinearAlgebra.QRCompactWY(zeros(T,0,0),zeros(T,0,0))
end

function qr_instance(A::Matrix{BigFloat})
    LinearAlgebra.QR(zeros(BigFloat,0,0),zeros(BigFloat,0))
end

# Could be optimized but this should work for any real case.
function qr_instance(jac_prototype::SparseMatrixCSC)
    qr(sparse(rand(1,1)))
end

"""
  qr_instance(a::Number) -> a

Returns the number.
"""
qr_instance(a::Number) = a

"""
    qr_instance(a::Any) -> qr(a)

Slow fallback which gets the instance via factorization. Should get
specialized for new matrix types.
"""
qr_instance(a::Any) = qr(a)# check = false)

"""
  svd_instance(A) -> qr_factorization_instance

Returns an instance of the SVD factorization object with the correct type
cheaply.
"""
function svd_instance(A::Matrix{T}) where {T}
    LinearAlgebra.SVD(zeros(T,0,0),zeros(T,0),zeros(T,0,0))
end

"""
  svd_instance(a::Number) -> a

Returns the number.
"""
svd_instance(a::Number) = a

"""
    svd_instance(a::Any) -> svd(a)

Slow fallback which gets the instance via factorization. Should get
specialized for new matrix types.
"""
svd_instance(a::Any) = svd(a) #check = false)

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
    undefmatrix(u::AbstractVector)

Creates the matrix version of `u` with possibly undefined values. Note that this is unique because
`similar(u,length(u),length(u))` returns a mutable type, so it is not type-matching,
while `fill(zero(eltype(u)),length(u),length(u))` doesn't match the array type,
i.e., you'll get a CPU array from a GPU array. The generic fallback is
`u .* u'`, which works on a surprising number of types, but can be broken
with weird (recursive) broadcast overloads. For higher-order tensors, this
returns the matrix linear operator type which acts on the `vec` of the array.
"""
function undefmatrix(u)
    similar(u, length(u), length(u))
end
function undefmatrix(u::Number)
    return zero(u)
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
device(::Type{T}) where {T <: Array} = CPUPointer()
device(::Type{T}) where {T <: AbstractArray} = _device(parent_type(T), T)
function _device(::Type{P}, ::Type{T}) where {P, T}
    if defines_strides(T)
        return device(P)
    else
        return _not_pointer(device(P))
    end
end
_not_pointer(::CPUPointer) = CPUIndex()
_not_pointer(x) = x
_device(::Type{T}, ::Type{T}) where {T <: DenseArray} = CPUPointer()
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
the ability to iterate (although additional functionality may be provided for specific
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

Base.firstindex(i::Union{BidiagonalIndex, TridiagonalIndex}) = 1
Base.lastindex(i::Union{BidiagonalIndex, TridiagonalIndex}) = i.count
Base.length(i::Union{BidiagonalIndex, TridiagonalIndex}) = lastindex(i)

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


"""
    ndims_index(::Type{I}) -> Int

Returns the number of dimensions that an instance of `I` indexes into. If this method is
not explicitly defined, then `1` is returned.

See also [`ndims_shape`](@ref)

# Examples

```julia
julia> ArrayInterface.ndims_index(Int)
1

julia> ArrayInterface.ndims_index(CartesianIndex(1, 2, 3))
3

julia> ArrayInterface.ndims_index([CartesianIndex(1, 2), CartesianIndex(1, 3)])
2

```
"""
ndims_index(::Type{<:Base.AbstractCartesianIndex{N}}) where {N} = N
# preserve CartesianIndices{0} as they consume a dimension.
ndims_index(::Type{CartesianIndices{0, Tuple{}}}) = 1
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
julia> ArrayInterface.ndims_shape([CartesianIndex(1, 1), CartesianIndex(1, 2)])
1

julia> ndims(CartesianIndices((2,2))[[CartesianIndex(1, 1), CartesianIndex(1, 2)]])
1

"""
ndims_shape(T::DataType) = ndims_index(T)
ndims_shape(::Type{Colon}) = 1
ndims_shape(@nospecialize T::Type{<:CartesianIndices}) = ndims(T)
ndims_shape(@nospecialize T::Type{<:Union{Number, Base.AbstractCartesianIndex}}) = 0
ndims_shape(@nospecialize T::Type{<:AbstractArray{Bool}}) = 1
ndims_shape(@nospecialize T::Type{<:AbstractArray}) = ndims(T)
ndims_shape(x) = ndims_shape(typeof(x))



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
indices_do_not_alias(::Type{T}) where {T} = _indices_do_not_alias(T)
function indices_do_not_alias(::Type{A}) where {T, A <: Base.StridedArray{T}}
    instances_do_not_alias(T)
end
function indices_do_not_alias(::Type{Adjoint{T, A}}) where {T, A <: AbstractArray{T}}
    indices_do_not_alias(A)
end
function indices_do_not_alias(::Type{Transpose{T, A}}) where {T, A <: AbstractArray{T}}
    indices_do_not_alias(A)
end
_indices_do_not_alias(::Type) = false
function _indices_do_not_alias(::Type{<:SubArray{<:Any, <:Any, A, I}}) where
    {
     A,
     I <: Tuple{
           Vararg{
                  Union{Integer, UnitRange, Base.ReshapedUnitRange,
                        Base.AbstractCartesianIndex}}}}
    indices_do_not_alias(A)
end

"""
    defines_strides(::Type{T}) -> Bool

Is strides(::T) defined? It is assumed that types returning `true` also return a valid
pointer on `pointer(::T)`.
"""
defines_strides(x) = defines_strides(typeof(x))
_defines_strides(::Type{T}, ::Type{T}) where {T} = false
_defines_strides(::Type{P}, ::Type{T}) where {P, T} = defines_strides(P)
defines_strides(::Type{T}) where {T} = _defines_strides(parent_type(T), T)
defines_strides(@nospecialize T::Type{<:StridedArray}) = true
defines_strides(@nospecialize T::Type{<:BitArray}) = true
@inline function defines_strides(@nospecialize T::Type{<:SubArray})
    stride_preserving_index(fieldtype(T, :indices))
end

#=
    stride_preserving_index(::Type{T}) -> Bool

Returns `True` if strides between each element can still be derived when indexing with an
instance of type `T`.
=#
stride_preserving_index(@nospecialize T::Type{<:AbstractRange}) = true
stride_preserving_index(@nospecialize T::Type{<:Number}) = true
@inline function stride_preserving_index(@nospecialize T::Type{<:Tuple})
    all(map_tuple_type(stride_preserving_index, T))
end
stride_preserving_index(@nospecialize T::Type) = false

## Stubs
struct BandedMatrixIndex <: ArrayInterface.MatrixIndex
    count::Int
    rowsize::Int
    colsize::Int
    bandinds::Array{Int,1}
    bandsizes::Array{Int,1}
    isrow::Bool
end

"""
    ensures_all_unique(T::Type) -> Bool

Returns `true` if all instances of type `T` are composed of a unique set of elements.
This does not require that `T` subtypes `AbstractSet` or implements the `AbstractSet`
interface.

# Examples

```julia
julia> ArrayInterface.ensures_all_unique(BitSet())
true

julia> ArrayInterface.ensures_all_unique([])
false

julia> ArrayInterface.ensures_all_unique(typeof(1:10))
true

julia> ArrayInterface.ensures_all_unique(LinRange(1, 1, 10))
false
```
"""
ensures_all_unique(@nospecialize T::Type{<:Union{AbstractSet,AbstractDict}}) = true
ensures_all_unique(@nospecialize T::Type{<:LinRange}) = false
ensures_all_unique(@nospecialize T::Type{<:AbstractRange}) = true
@inline function ensures_all_unique(T::Type)
    is_forwarding_wrapper(T) ? ensures_all_unique(parent_type(T)) : false
end
ensures_all_unique(@nospecialize(x)) = ensures_all_unique(typeof(x))

"""
    ensures_sorted(T::Type) -> Bool

Returns `true` if all instances of `T` are sorted.

# Examples

```julia
julia> ArrayInterface.ensures_sorted(BitSet())
true

julia> ArrayInterface.ensures_sorted([])
false

julia> ArrayInterface.ensures_sorted(1:10)
true
```
"""
ensures_sorted(@nospecialize(T::Type{BitSet})) = true
ensures_sorted(@nospecialize( T::Type{<:AbstractRange})) = true
ensures_sorted(T::Type) = is_forwarding_wrapper(T) ? ensures_sorted(parent_type(T)) : false
ensures_sorted(@nospecialize(x)) = ensures_sorted(typeof(x))

## Extensions

import Requires
@static if !isdefined(Base, :get_extension)
    function __init__()
        Requires.@require BandedMatrices = "aae01518-5342-5314-be14-df237901396f" begin include("../ext/ArrayInterfaceBandedMatricesExt.jl") end
        Requires.@require BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0" begin include("../ext/ArrayInterfaceBlockBandedMatricesExt.jl") end
        Requires.@require GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527" begin include("../ext/ArrayInterfaceGPUArraysCoreExt.jl") end
        Requires.@require StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c" begin include("../ext/ArrayInterfaceStaticArraysCoreExt.jl") end
        Requires.@require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin include("../ext/ArrayInterfaceCUDAExt.jl") end
        Requires.@require Tracker="9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin include("../ext/ArrayInterfaceTrackerExt.jl") end
    end
end

end # module
