module ArrayInterfaceGPUArraysCoreTrackerExt

using Adapt
using ArrayInterface
import GPUArraysCore
import Tracker

# Tracker.adapt_structure uses `param(adapt(T, data(xs)))`, which severs the tape.
function _adapt_tracked_storage(T, y)
    Adapt.adapt(T, y)
end
function _adapt_tracked_storage(T, y::Tracker.TrackedArray)
    Tracker.track(_adapt_tracked_storage, T, y)
end
Tracker.@grad function _adapt_tracked_storage(T, y)
    ydata = Tracker.data(y)
    Adapt.adapt(T, ydata),
    Δ -> (nothing, Adapt.adapt(ArrayInterface.parameterless_type(ydata), Tracker.data(Δ)))
end

function ArrayInterface.restructure(
        x::GPUArraysCore.AbstractGPUArray, y::Tracker.TrackedArray)
    T = ArrayInterface.parameterless_type(x)
    yT = Tracker.data(y) isa T ? y : _adapt_tracked_storage(T, y)
    reshape(yT, Base.size(x)...)
end

end
