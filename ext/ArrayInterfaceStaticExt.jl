module ArrayInterfaceStaticExt

if isdefined(Base, :get_extension)
    import ArrayInterface
    import Static
else
    import ..ArrayInterface
    import ..Static
end

ArrayInterface.known_first(::Type{<:Static.OptionallyStaticUnitRange{Static.StaticInt{F}}}) where {F} = F::Int
ArrayInterface.known_first(::Type{<:Static.OptionallyStaticStepRange{Static.StaticInt{F}}}) where {F} = F::Int

ArrayInterface.known_step(::Type{<:Static.OptionallyStaticStepRange{<:Any,Static.StaticInt{S}}}) where {S} = S::Int

ArrayInterface.known_last(::Type{<:Static.OptionallyStaticUnitRange{<:Any,Static.StaticInt{L}}}) where {L} = L::Int
ArrayInterface.known_last(::Type{<:Static.OptionallyStaticStepRange{<:Any,<:Any,Static.StaticInt{L}}}) where {L} = L::Int

end
