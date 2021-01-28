# ArrayInterface.jl

[![CI](https://github.com/SciML/ArrayInterface.jl/workflows/CI/badge.svg)](https://github.com/SciML/ArrayInterface.jl/actions?query=workflow%3ACI)
[![CI (Julia nightly)](https://github.com/SciML/ArrayInterface.jl/workflows/CI%20(Julia%20nightly)/badge.svg)](https://github.com/SciML/ArrayInterface.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22)
[![Build status](https://badge.buildkite.com/a2db252d92478e1d7196ee7454004efdfb6ab59496cbac91a2.svg?branch=master)](https://buildkite.com/julialang/arrayinterface-dot-jl)
[![codecov](https://codecov.io/gh/SciML/ArrayInterface.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/ArrayInterface.jl)

The AbstractArray interface in Base Julia is still relatively young.
The purpose of this library is to solidify extensions to the current
AbstractArray interface, which are put to use in package ecosystems like
DifferentialEquations.jl. Since these libraries are live, this package will
serve as a staging ground for ideas before they are merged into Base Julia. For this
reason, no functionality is exported so that if such functions are added
and exported in a future Base Julia, there will be no issues with the upgrade.

## parent_type(x)

Returns the parent array that `x` wraps.

## can_change_size(x)

Returns `true` if the size of `T` can change, in which case operations
such as `pop!` and `popfirst!` are available for collections of type `T`.

## indices(x[, d])

Given an array `x`, this returns the indices along dimension `d`. If `x` is a tuple
of arrays, then the indices corresponding to dimension `d` of all arrays in `x` are
returned. If any indices are not equal along dimension `d`, an error is thrown. A
tuple may be used to specify a different dimension for each array. If `d` is not
specified, then indices for visiting each index of `x` are returned.

## ismutable(x)

A trait function for whether `x` is a mutable or an immutable array. Used for
dispatching to in-place and out-of-place versions of functions.

## aos_to_soa(x)

Converts an array of structs formulation to a struct of arrays.

## isstructured(x)

A trait function for whether a matrix `x` is a sparse structured matrix.

## can_setindex(x)

A trait function for whether an array `x` can use `setindex!`.

## has_sparsestruct(x)

Determine whether `findstructralnz` accepts the parameter `x`.

## findstructralnz(x)

Returns iterators `(I,J)` of the non-zeros in the structure of the matrix `x`.
The same as the first two elements of `findnz(::SparseMatrixCSC)`.

## fast_matrix_colors(A)

A trait function for whether `matrix_colors(A)` is a fast algorithm or a slow
(graphically-based) method.

## matrix_colors(A)

Returns an array for the sparsity colors of a matrix type `A`. Also includes
an abstract type `ColoringAlgorithm` for `matrix_colors(A,alg::ColoringAlgorithm)`
of non-structured matrices.

## fast_scalar_indexing(A)

A trait function for whether scalar indexing is fast on a given array type.

## allowed_getindex(A,i...)

A `getindex` which is always allowed.

## allowed_setindex!(A,v,i...)

A `setindex!` which is always allowed.

## lu_instance(A)

Returns an instance of the LU factorization object with the correct type
cheaply.

## issingular(A)

Returns an instance of the LU factorization object with the correct type cheaply.

## safevec(v)

Is a form of `vec` which is safe for all values in vector spaces, i.e., if it
is already a vector, like an AbstractVector or Number, it will return said
AbstractVector or Number.

## zeromatrix(u)

Creates the zero'd matrix version of `u`. Note that this is unique because
`similar(u,length(u),length(u))` returns a mutable type, so is not type-matching,
while `fill(zero(eltype(u)),length(u),length(u))` doesn't match the array type,
i.e., you'll get a CPU array from a GPU array. The generic fallback is
`u .* u' .* false`, which works on a surprising number of types, but can be broken
with weird (recursive) broadcast overloads. For higher-order tensors, this
returns the matrix linear operator type, which acts on the `vec` of the array.

## restructure(x,y)

Restructures the object `y` into a shape of `x`, keeping its values intact. For
simple objects, like an `Array`, this simply amounts to a reshape. However, for
more complex objects such as an `ArrayPartition`, not all of the structural
information is adequately contained in the type for standard tools to work. In
these cases, `restructure` gives a way to convert, for example, an `Array` into
a matching `ArrayPartition`.

## known_first(::Type{T})

If `first` of instances of type `T` are known at compile time, return that first
element. Otherwise, return `nothing`. For example, `known_first(Base.OneTo{Int})`
returns `one(Int)`.

## known_last(::Type{T})

If `last` of instances of type `T` are known at compile time, return that
last element. Otherwise, return `nothing`. For example,
`known_last(StaticArrays.SOneTo{4})` returns 4.

## known_step(::Type{T})

If `step` of instances of type `T` are known at compile time, return that step.
Otherwise, returns `nothing`. For example, `known_step(UnitRange{Int})` returns
`one(Int)`.

## known_length(::Type{T})

If `length` of an instance of type `T` is known at compile time, return it.
Otherwise, return `nothing`.

## device(::Type{T})

Indicates the most efficient way to access elements from the collection in low-level code.
For `GPUArrays`, returns `ArrayInterface.GPU()`.
For `AbstractArray` supporting a `pointer` method, returns `ArrayInterface.CPUPointer()`.
For other `AbstractArray`s and `Tuple`s, returns `ArrayInterface.CPUIndex()`.
Otherwise, returns `nothing`.

## contiguous_axis(::Type{T})

Returns the axis of an array of type `T` containing contiguous data.
If no axis is contiguous, it returns `Contiguous{-1}`.
If unknown, it returns `nothing`.

## contiguous_axis_indicator(::Type{T})

Returns a tuple of boolean `Val`s indicating whether that axis is contiguous.

## contiguous_batch_size(::Type{T})

Returns the size of contiguous batches if `!isone(stride_rank(T, contiguous_axis(T)))`.
If `isone(stride_rank(T, contiguous_axis(T)))`, then returns `ContiguousBatch{0}()`.
If `contiguous_axis(T) == -1`, returns `ContiguousBatch{-1}()`.
If unknown, returns `nothing`.

## stride_rank(::Type{T})

Returns the rank of each stride.

## is_column_major(A)

Returns a `Val{true}()` if `A` is column major, and a `Val{false}()` otherwise.`

## dense_dims(::Type{T})
Returns a tuple of indicators for whether each axis is dense.
An axis `i` of array `A` is dense if `stride(A, i) * size(A, i) == stride(A, j)`, where `j` is the axis (if it exists) such that `stride_rank(A)[i] + 1 == stride_rank(A)[j]`.

## ArrayInterface.size(A)

Returns the size of `A`. If the sizes of any axes are known at compile time,
these should be returned as `StaticInt`s. For example:
```julia
julia> using StaticArrays, ArrayInterface

julia> A = @SMatrix rand(3,4);

julia> ArrayInterface.size(A)
(static(3), static(4))
```

## ArrayInterface.strides(A)

Returns the strides of array `A`. If any strides are known at compile time,
these should be returned as `StaticInt`s. For example:
```julia
julia> using ArrayInterface

julia> A = rand(3,4);

julia> ArrayInterface.strides(A)
(static(1), 3)
```
## offsets(A)

Returns offsets of indices with respect to 0. If values are known at compile time,
it should return them as `StaticInt`s.
For example, if `A isa Base.Matrix`, `offsets(A) === (StaticInt(1), StaticInt(1))`.

## can_avx(f)

Is the function `f` whitelisted for `LoopVectorization.@avx`?

## StaticInt(N::Int)

Creates a static integer with value known at compile time. It is a number,
supporting basic arithmetic. Many operations with two `StaticInt` integers
will produce another `StaticInt` integer. If one of the arguments to a
function call isn't static (e.g., `StaticInt(4) + 3`), then the `StaticInt`
number will promote to a dynamic value.

# List of things to add

- https://github.com/JuliaLang/julia/issues/22216
- https://github.com/JuliaLang/julia/issues/22218
- https://github.com/JuliaLang/julia/issues/22622
- https://github.com/JuliaLang/julia/issues/25760
- https://github.com/JuliaLang/julia/issues/25107

# Array Types to Handle

The following common array types are being understood and tested as part of this
development.

- Array
- Various versions of sparse arrays
- SArray
- MArray
- FieldVector
- ArrayPartition
- VectorOfArray
- DistributedArrays
- GPUArrays (CLArrays and CuArrays)
- AFArrays
- MultiScaleArrays
- LabelledArrays

## Breaking Release Notes

2.0: Changed the default of `ismutable(array::AbstractArray) = true`. We previously defaulted to
`Base.@pure ismutable(array::AbstractArray) = typeof(array).mutable`, but there are a lot of cases
where this tends to not work out in a way one would expect. For example, if you put a normal array
into an immutable struct that adds more information to it, this is considered immutable, even if
all of the `setindex!` methods work (by forwarding to the mutable array). Thus, it seems safer to just
always assume mutability is standard for an array, and allow arrays to opt-out.
