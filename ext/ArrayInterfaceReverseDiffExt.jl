module ArrayInterfaceReverseDiffExt

using ArrayInterface
import ReverseDiff

ArrayInterface.ismutable(::Type{<:ReverseDiff.TrackedArray}) = false
ArrayInterface.ismutable(T::Type{<:ReverseDiff.TrackedReal}) = false
ArrayInterface.can_setindex(::Type{<:ReverseDiff.TrackedArray}) = false
ArrayInterface.fast_scalar_indexing(::Type{<:ReverseDiff.TrackedArray}) = false
function ArrayInterface.aos_to_soa(x::AbstractArray{<:ReverseDiff.TrackedReal, N}) where {N}
    y = length(x) > 1 ? reduce(vcat, x) : reduce(vcat, [x[1], x[1]])[1:1]
    return reshape(y, size(x))
end

function ArrayInterface.restructure(x::Array, y::ReverseDiff.TrackedArray)
    reshape(y, Base.size(x)...)
end

end # module
