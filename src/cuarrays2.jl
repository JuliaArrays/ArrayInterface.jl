fast_scalar_indexing(::Type{<:CUDA.CuArray}) = false
@inline allowed_getindex(x::CUDA.CuArray, i...) = CUDA.@allowscalar(x[i...])
@inline allowed_setindex!(x::CUDA.CuArray, v, i...) = (CUDA.@allowscalar(x[i...] = v))

function Base.setindex(x::CUDA.CuArray, v, i::Int)
    _x = copy(x)
    allowed_setindex!(_x, v, i)
    return _x
end

function restructure(x::CUDA.CuArray, y)
    return reshape(Adapt.adapt(parameterless_type(x), y), Base.size(x)...)
end

device(::Type{<:CUDA.CuArray}) = GPU()
issparsematrix(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = true
