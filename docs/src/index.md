```@meta
CurrentModule = ArrayInterface
```

# ArrayInterface

```@index
```

```@autodocs
Modules = [ArrayInterface]
```

## Static Traits

The size along one or more dimensions of an array may be know at compile time. 
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

Generally, `ArrayInterface.size` uses the return of `known_size` to form a static value for those dimensions with known length and only queries dimensions corresponding to `nothing`.
For example, the previous example had a known size of `(1, nothing)`.
Therefore, `ArrayInterface.size` would have compile time information about the first dimension returned as `static(1)` and would only look up the size of the second dimension at run time.
Generic support for `ArrayInterface.known_size` relies on calling `known_length` for each type returned from `axes_types`.
Therefore, the recommended approach for supporting static sizing in newly defined array types is defining a new `axes_types` method.

Static information related to subtypes of `AbstractRange` include `known_length`, `known_first`, `known_step`, and `known_last`.

