
""" BroadcastAxis """
abstract type BroadcastAxis end

struct BroadcastAxisDefault <: BroadcastAxis end

BroadcastAxis(x) = BroadcastAxis(typeof(x))
BroadcastAxis(::Type{T}) where {T} = BroadcastAxisDefault()

""" broadcast_axis(x, y) """
broadcast_axis(x, y) = broadcast_axis(BroadcastAxis(x), x, y)
# stagger default broadcasting in case y has something other than default
broadcast_axis(::BroadcastAxisDefault, x, y) = _broadcast_axis(BroadcastAxis(y), x, y)
function _broadcast_axis(::BroadcastAxisDefault, x, y)
    return One():_combine_length(static_length(x), static_length(y))
end
_broadcast_axis(s::BroadcastAxis, x, y) = broadcasted_axis(s, x, y)
_combine_length(::StaticInt{X}, ::StaticInt{Y}) where {X,Y} = static(_combine_length(X, Y))
_combine_length(::StaticInt{X}, y::Int) where {X} = _combine_length(X, y)
_combine_length(x::Int, ::StaticInt{Y}) where {Y} = _combine_length(x, Y)
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
