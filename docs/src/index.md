```@meta
CurrentModule = ArrayInterface
```

# ArrayInterface

Designs for new Base array interface primitives, used widely through scientific machine learning (SciML) and other organizations

## Inheriting Array Traits

Creating an array type with unique behavior in Julia is often accomplished by creating a lazy wrapper around previously defined array types.
This allows the new array type to inherit functionality by redirecting methods to the parent array (e.g., `Base.size(x::Wrapper) = size(parent(x))`).
Generic design limits the need to define an excessive number of methods like this.
However, methods used to describe a type's traits often need to be explicitly defined for each trait method.
`ArrayInterface` assists with this by providing information about the parent type using [`ArrayInterface.parent_type`](@ref).
By default `ArrayInterface.parent_type(::Type{T})` returns `T` (analogous to `Base.parent(x) = x`).
If any type other than `T` is returned we assume `T` wraps a parent structure, so methods know to unwrap instances of `T`.
It is also assumed that if `T` has a parent type `Base.parent` is defined.

For those authoring new trait methods, this may change the default definition from `has_trait(::Type{T}) where {T} = false`, to:
```julia
function has_trait(::Type{T}) where {T}
    if parent_type(T) <:T
        return false
    else
        return has_trait(parent_type(T))
    end
end
```

Most traits in `ArrayInterface` are a variant on this pattern.

## Static Traits

The size along one or more dimensions of an array may be known at compile time. 
`ArrayInterface.known_size` is useful for extracting this information from array types and `ArrayInterface.size` is useful for extracting this information from an instance of an array.
For example:

```julia
julia> a = ones(3)';

julia> ArrayInterface.size(a)
(static(1), 3)

julia> ArrayInterface.known_size(typeof(a))
(1, nothing)

```

This is useful for dispatching on known information about the size of an array:
```julia
fxn(x) = _fxn(ArrayInterface.size(x), x)
_fxn(sz::Tuple{StaticInt{S1},StaticInt{S2}}, x) where {S1,S2} = ...
_fxn(sz::Tuple{StaticInt{3},StaticInt{3}}, x) = ...
_fxn(sz::Tuple{Int,StaticInt{S2}}, x) where {S2} = ...
_fxn(sz::Tuple{StaticInt{S1},Int}, x) where {S1} = ...
_fxn(sz::Tuple{Int,Int}, x) = ...
```

Methods should avoid forcing conversion to static sizes when dynamic sizes could potentially be returned.
Fore example, `fxn(x) = _fxn(Static.static(ArrayInterface.size(x)), x)` would result in dynamic dispatch if `x` is an instance of `Matrix`.
Additionally, `ArrayInterface.size` should only be used outside of generated functions to avoid possible world age issues.

Generally, `ArrayInterface.size` uses the return of `known_size` to form a static value for those dimensions with known length and only queries dimensions corresponding to `nothing`.
For example, the previous example had a known size of `(1, nothing)`.
Therefore, `ArrayInterface.size` would have compile time information about the first dimension returned as `static(1)` and would only look up the size of the second dimension at run time.
This means the above example `ArrayInterface.size(a)` would lower to code similar to this at compile time: `Static.StaticInt(1), Base.arraysize(x, 1)`.
Generic support for `ArrayInterface.known_size` relies on calling `known_length` for each type returned from `axes_types`.
Therefore, the recommended approach for supporting static sizing in newly defined array types is defining a new `axes_types` method.

Static information related to subtypes of `AbstractRange` include `known_length`, `known_first`, `known_step`, and `known_last`.

## Dimensions

Methods such as `size(x, dim)` need to map `dim` to the dimensions of `x`.
Typically, `dim` is an `Int` with an invariant mapping to the dimensions of `x`.
Some methods accept `:` or a tuple of dimensions as an argument.
`ArrayInterface` also considers `StaticInt` a viable dimension argument.

[`ArrayInterface.to_dims`](@ref) helps ensure that `dim` is converted to a viable dimension mapping in a manner that helps with type stability.
For example, all `Integers` passed to `to_dims` are converted to `Int` (unless `dim` is a `StaticInt`).
This is also useful for arrays that uniquely label dimensions, in which case `to_dims` serves as a safe point of hooking into existing methods with dimension arguments.
`ArrayInterface` also defines native `Symbol` to `Int` and `StaticSymbol` to `StaticInt` mapping  for arrays defining [`ArrayInterface.dimnames`](@ref).

Methods accepting dimension specific arguments should use some variation of the following pattern.

```julia
f(x, dim) = f(x, ArrayInterface.to_dims(x, dim))
f(x, dim::Int) = ...
f(x, dim::StaticInt) = ...
```

If `x`'s first dimension is named `:dim_1` then calling `f(x, :dim_1)` would result in `f(x, 1)`.
If users knew they always wanted to call `f(x, 2)` then they could define `h(x) = f(x, static(2))`, ensuring `f` passes along that information while compiling.

