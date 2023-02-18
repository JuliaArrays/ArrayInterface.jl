struct BroadcastAxisDefault <: BroadcastAxis end

BroadcastAxis(x) = BroadcastAxis(typeof(x))
BroadcastAxis(::Type{T}) where {T} = BroadcastAxisDefault()

"""
    broadcast_axis(x, y)

Broadcast axis `x` and `y` into a common space. The resulting axis should be equal in length
to both `x` and `y` unless one has a length of `1`, in which case the longest axis will be
equal to the output.

```julia
julia> ArrayInterface.broadcast_axis(1:10, 1:10)

julia> ArrayInterface.broadcast_axis(1:10, 1)
1:10

```
"""
broadcast_axis(x, y) = broadcast_axis(BroadcastAxis(x), x, y)
# stagger default broadcasting in case y has something other than default
broadcast_axis(::BroadcastAxisDefault, x, y) = _broadcast_axis(BroadcastAxis(y), x, y)
function _broadcast_axis(::BroadcastAxisDefault, x, y)
    return One():_combine_length(static_length(x), static_length(y))
end
_broadcast_axis(s::BroadcastAxis, x, y) = broadcast_axis(s, x, y)

# we can use a similar trick as we do with `indices` where unequal sizes error and we just
# keep the static value. However, axes can be unequal if one of them is `1` so we have to
# fall back to dynamic values in those cases
_combine_length(x::StaticInt{X}, y::StaticInt{Y}) where {X,Y} = static(_combine_length(X, Y))
_combine_length(x::StaticInt{X}, ::Int) where {X} = x
_combine_length(x::StaticInt{1}, y::Int) = y
_combine_length(x::StaticInt{1}, y::StaticInt{1}) = y
_combine_length(x::Int, y::StaticInt{Y}) where {Y} = y
_combine_length(x::Int, y::StaticInt{1}) = x
@inline function _combine_length(x::Int, y::Int)
    if x === y
        return x
    elseif y === 1
        return x
    elseif x === 1
        return y
    else
        _dimerr(x, y)
    end
end

function _dimerr(@nospecialize(x), @nospecialize(y))
    throw(DimensionMismatch("axes could not be broadcast to a common size; " *
                            "got axes with lengths $(x) and $(y)"))
end
