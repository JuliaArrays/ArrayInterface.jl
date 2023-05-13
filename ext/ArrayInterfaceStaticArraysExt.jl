module ArrayInterfaceStaticArraysExt

if isdefined(Base, :get_extension)
    import ArrayInterface
    import StaticArrays
else
    import ..ArrayInterface
    import ..StaticArrays
end

ArrayInterface.known_first(@nospecialize T::Type{<:StaticArrays.SOneTo}) = 1
ArrayInterface.known_last(::Type{StaticArrays.SOneTo{N}}) where {N} = @isdefined(N) ? N::Int : nothing

function ArrayInterface.known_first(::Type{<:StaticArrays.SUnitRange{S}}) where {S}
    @isdefined(S) ? S::Int : nothing
end
function ArrayInterface.known_size(::Type{<:StaticArrays.SUnitRange{<:Any, L}}) where {L}
    @isdefined(L) ? (L::Int,) : (nothing,)
end
function ArrayInterface.known_last(::Type{<:StaticArrays.SUnitRange{S, L}}) where {S, L}
    start = @isdefined(S) ? S::Int : nothing
    len = @isdefined(L) ? L::Int : nothing
    (start === nothing || len === nothing) ? nothing : (start + len - 1)
end

end
