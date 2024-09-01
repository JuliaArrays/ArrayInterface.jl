module ArrayInterfaceTrackerExt

using ArrayInterface
import Tracker

ArrayInterface.ismutable(::Type{<:Tracker.TrackedArray}) = false
ArrayInterface.ismutable(T::Type{<:Tracker.TrackedReal}) = false
ArrayInterface.can_setindex(::Type{<:Tracker.TrackedArray}) = false
ArrayInterface.fast_scalar_indexing(::Type{<:Tracker.TrackedArray}) = false
ArrayInterface.aos_to_soa(x::AbstractArray{<:Tracker.TrackedReal,N}) where {N} = Tracker.collect(x)

function ArrayInterface.restructure(x::Array, y::Tracker.TrackedArray)
    reshape(y, Base.size(x)...)
end
function ArrayInterface.restructure(x::Array, y::Array{<:Tracker.TrackedReal})
    reshape(y, Base.size(x)...)
end

end # module
