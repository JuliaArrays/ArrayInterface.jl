module ArrayInterfaceGPUArraysCoreTrackerExt

using ArrayInterface
import GPUArraysCore
import Tracker

# Tracker.adapt_structure uses `param(adapt(T, data(xs)))`, which severs the tape.
function ArrayInterface.restructure(
        x::GPUArraysCore.AbstractGPUArray, y::Tracker.TrackedArray)
    reshape(y, Base.size(x)...)
end

end
