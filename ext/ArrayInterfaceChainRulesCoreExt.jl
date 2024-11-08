module ArrayInterfaceChainRulesCoreExt

import ArrayInterface
import ChainRulesCore
import ChainRulesCore: unthunk, NoTangent, ZeroTangent, ProjectTo, @thunk

function ChainRulesCore.rrule(::typeof(ArrayInterface.restructure), target, src)
    projectT = ProjectTo(target)
    function restructure_pullback(dt)
        dt = unthunk(dt)

        f̄ = NoTangent()
        t̄ = ZeroTangent()
        s̄ = @thunk(projectT(ArrayInterface.restructure(src, dt)))

        f̄, t̄, s̄
    end

    return ArrayInterface.restructure(target, src), restructure_pullback
end

end
