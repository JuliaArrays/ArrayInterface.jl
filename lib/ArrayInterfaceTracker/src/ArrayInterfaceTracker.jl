module ArrayInterfaceTracker

using ArrayInterfaceCore
using Tracker

ArrayInterfaceCore.ismutable(::Type{<:Tracker.TrackedArray}) = false
ArrayInterfaceCore.ismutable(T::Type{<:Tracker.TrackedReal}) = false
ArrayInterfaceCore.can_setindex(::Type{<:Tracker.TrackedArray}) = false
ArrayInterfaceCore.fast_scalar_indexing(::Type{<:Tracker.TrackedArray}) = false
ArrayInterfaceCore.aos_to_soa(x::AbstractArray{<:Tracker.TrackedReal,N}) where {N} = Tracker.collect(x)

end # module
