module ArrayInterfaceChainRulesCoreExt

import ArrayInterface
import ChainRulesCore
import ChainRulesCore: unthunk, NoTangent, ZeroTangent, ProjectTo, @thunk

function ChainRulesCore.rrule(::typeof(ArrayInterface.restructure), target, src)
    projectS = ProjectTo(src)
    function restructure_pullback(dt)
        dt = unthunk(dt)

        f̄ = NoTangent()
        t̄ = ZeroTangent()
        s̄ = @thunk(projectS(ArrayInterface.restructure(src, dt)))

        f̄, t̄, s̄
    end

    return ArrayInterface.restructure(target, src), restructure_pullback
end

end
