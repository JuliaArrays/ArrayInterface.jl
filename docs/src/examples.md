# Examples

## Simple Array Wrapper

```julia
using ArrayInterface
using ArrayInterface: defines_strides, parent_type

struct Wrapper{T,N,P<:AbstractArray{T,N}} <: ArrayInterface.AbstractArray2{T,N}
    parent::P
end

ArrayInterface.parent_type(::Type{<:Wrapper{T,N,P}}) where {T,N,P} = P
Base.parent(x::Wrapper) = x.parent

ArrayInterface.defines_strides(::Type{T}) where {T<:Wrapper} = defines_strides(parent_type(T))
```

