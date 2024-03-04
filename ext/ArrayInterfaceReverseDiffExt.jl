module ArrayInterfaceReverseDiffExt

if isdefined(Base, :get_extension)
    using ArrayInterface
    import ReverseDiff
else
    using ..ArrayInterface
    import ..ReverseDiff
end

ArrayInterface.ismutable(::Type{<:ReverseDiff.TrackedArray}) = false
ArrayInterface.ismutable(T::Type{<:ReverseDiff.TrackedReal}) = false
ArrayInterface.can_setindex(::Type{<:ReverseDiff.TrackedArray}) = false
ArrayInterface.fast_scalar_indexing(::Type{<:ReverseDiff.TrackedArray}) = false
function ArrayInterface.aos_to_soa(x::AbstractArray{<:ReverseDiff.TrackedReal,N}) where {N}
    if length(x) > 1
        reduce(vcat,x)
    else
        @show "here?"
        reduce(vcat,[x[1],x[1]])[1:1]
    end
end

end # module
