Designs for new Base array interface primitives, used widely through scientific machine learning (SciML) and other organizations

## Inheriting Array Traits

A common design pattern in Julia is to inherit behaviors of a type by simply wrapping it in a new type.
This can result in a bit of boiler plate (e.g., `Base.size(x::Wrapper) = size(parent(x))`).
`ArrayInterface` assists with this by specifying the parent at the type level using [`ArrayInterface.parent_type`](@ref).
By default `ArrayInterface.parent_type(::Type{T})` returns `T` (similar to how `Base.parent(x) = x`).
If any type other than `T` is returned we assume `T` wraps a parent structure, so methods know to unwrap instances of `T`.
It is also assumed that if `T` has a parent type `Base.parent` is defined.
Inheriting traits also becomes fairly simple with this approach.
For example, the following will return `true` even if `SpecialArray` deeply nested within `T`.

```julia
has_trait(::Type{T}) where {T<:SpecialArray} = true
function has_trait(::Type{T}) where {T}
    if parent_type(T) <:T
        return false
    else
        return has_trait(parent_type(T))
    end
end
```

For methods where `f(x) = f(parent(x))` is appropriate, this simplifies the process of defining new array types and writing generic methods.
Much of `ArrayInterface` is dedicated to defining generic methods and well defined entry points when `f(x) = f(parent(x))` isn't appropriate.
The following sections elaborate on this for some of the more common aspects of working with arrays (dimensions, indexing, memory layout, etc.).

## Dimensions

Methods such as `size(x, dim)` need to map `dim` to the dimensions of `x`.
Typically, `dim` is an `Int` with an invariant mapping to the dimensions of `x`.
Some methods accept `:` or a tuple of dimensions as an argument.
`ArrayInterface` also considers `StaticInt` a viable dimension argument.

[`ArrayInterface.to_dims`](@ref) helps ensure that `dim` is converted to a viable dimension mapping type stable argument.
For example, all `Integers` passed to `to_dims` are converted to an `Int` (unless `dim` is a `StaticInt`).
This is also useful for arrays that uniquely label dimensions, in which case `to_dims` serves as a safe point of hooking into existing methods with dimension arguments.
`ArrayInterface` also defines `Symbol` to `Int` mapping natively for arrays defining [`ArrayInterface.dimnames`](@ref).

### Dimension-wise Methods

Most methods accepting dimension specific arguments can reliably use the following pattern.

```julia
f(x, dim) = g(x, ArrayInterface.to_dims(x, dim))
```

If `x` has a dimension named `:dim_1` then calling `f(x, :dim_1)` would result in `g(x, 1)`.
If users knew they always wanted to call `f(x, 2)` then they could define `h(x) = f(x, static(2))`, ensuring `g` passes along that information while compiling.
This also helps if `x` defines its own custom dimension name types.
For example...

```julia

struct FooBarMatrix{T}
    parent::Matrix{T}
end

abstract type FooBar end

struct Foo <: FooBar end

struct Bar <: FooBar end

ArrayInterface.dimnames(::Type{T}) where {T<:FooBarMatrix} = (Foo(), Bar())

function ArrayInterface.to_dims(::Type{T}, x::FooBar) where {T<:FooBarMatrix}
    return findfirst(==(x), dimnames(T))
end
```



### Mapping Dimensions

We typically assume that if an n-dimensional array wraps another array the mapping between dimensions is invariant.
This assumption is violated if dimensions are permuted, added, or dropped.
[`ArrayInterface.to_parent_dims`](@ref) and [`ArrayInterface.from_parent_dims`](@ref) provide a common interface for mapping between dimensions.


