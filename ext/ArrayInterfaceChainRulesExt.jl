module ArrayInterfaceChainRulesExt

using ArrayInterface
using ChainRules: OneElement

ArrayInterface.can_setindex(::Type{<:OneElement}) = false

end