# Examples

## Simple Array Wrapper

```julia
struct Wrapper{T,N,P<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::P
end

ArrayInterface.parent_type(::Type{<:Wrapper{T,N,P}}) where {T,N,P} = P

Base.parent(x::Wrapper) = x.parent

function ArrayInterface.unsafe_set_element!(x::Wrapper, val, inds)
    return ArrayInterface.unsafe_set_element!(parent(x), val, inds)
end

function ArrayInterface.unsafe_get_element(x::Wrapper, inds)
    return ArrayInterface.unsafe_get_element(parent(x), inds)
end

function Base.getindex(x::Wrapper, args...; kwargs...)
    return ArrayInterface.getindex(x, args...; kwargs...)
end
function Base.setindex!(x::Wrapper, val, args...; kwargs...)
    return ArrayInterface.setindex!(x, val, args...; kwargs...)
end

```



