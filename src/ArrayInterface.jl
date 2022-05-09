module ArrayInterface

using ArrayInterfaceCore
import ArrayInterfaceCore: axes, axes_types, can_setindex, contiguous_axis, contiguous_batch_size,
    defines_strides, dense_dims, device, dimnames, fast_scalar_indexing, findstructralnz,
    is_lazy_conjugate, length,
    has_sparsestruct, lu_instance, matrix_colors, ismutable, restructure, known_first,
    known_last, known_length, known_step, known_size, known_strides, known_offsets, offsets,
    parent_type, size, strides, stride_rank, to_dims, to_indices, to_index, zeromatrix
using Requires
using Static
using Static: Zero, One, nstatic, eq, ne, gt, ge, lt, le, eachop, eachop_tuple,
    permute, invariant_permutation, field_type, reduce_tup

function __init__()
    @require StaticArrays = "90137ffa-7385-5640-81b9-e52037218182" begin
        ismutable(::Type{<:StaticArrays.StaticArray}) = false
        can_setindex(::Type{<:StaticArrays.StaticArray}) = false
        ismutable(::Type{<:StaticArrays.MArray}) = true
        ismutable(::Type{<:StaticArrays.SizedArray}) = true

        buffer(A::Union{StaticArrays.SArray,StaticArrays.MArray}) = getfield(A, :data)

        function lu_instance(_A::StaticArrays.StaticMatrix{N,N}) where {N}
            A = StaticArrays.SArray(_A)
            L = LowerTriangular(A)
            U = UpperTriangular(A)
            p = StaticArrays.SVector{N,Int}(1:N)
            return StaticArrays.LU(L, U, p)
        end

        function restructure(x::StaticArrays.SArray, y::StaticArrays.SArray)
            reshape(y, StaticArrays.Size(x))
        end
        restructure(x::StaticArrays.SArray{S}, y) where {S} = StaticArrays.SArray{S}(y)

        known_first(::Type{<:StaticArrays.SOneTo}) = 1
        known_last(::Type{StaticArrays.SOneTo{N}}) where {N} = N
        known_length(::Type{StaticArrays.SOneTo{N}}) where {N} = N
        known_length(::Type{StaticArrays.Length{L}}) where {L} = L
        known_length(::Type{A}) where {A <: StaticArrays.StaticArray} = known_length(StaticArrays.Length(A))

        device(::Type{<:StaticArrays.MArray}) = CPUPointer()
        device(::Type{<:StaticArrays.SArray}) = CPUTuple()
        contiguous_axis(::Type{<:StaticArrays.StaticArray}) = StaticInt{1}()
        contiguous_batch_size(::Type{<:StaticArrays.StaticArray}) = StaticInt{0}()
        function stride_rank(::Type{T}) where {N,T<:StaticArrays.StaticArray{<:Any,<:Any,N}}
            Static.nstatic(Val(N))
        end
        function dense_dims(::Type{<:StaticArrays.StaticArray{S,T,N}}) where {S,T,N}
            return ArrayInterface._all_dense(Val(N))
        end
        defines_strides(::Type{<:StaticArrays.SArray}) = true
        defines_strides(::Type{<:StaticArrays.MArray}) = true

        @generated function axes_types(::Type{<:StaticArrays.StaticArray{S}}) where {S}
            return Tuple{[StaticArrays.SOneTo{s} for s in S.parameters]...}
        end
        @generated function size(A::StaticArrays.StaticArray{S}) where {S}
            t = Expr(:tuple)
            Sp = S.parameters
            for n = 1:length(Sp)
                push!(t.args, Expr(:call, Expr(:curly, :StaticInt, Sp[n])))
            end
            return t
        end
        @generated function strides(A::StaticArrays.StaticArray{S}) where {S}
            t = Expr(:tuple, Expr(:call, Expr(:curly, :StaticInt, 1)))
            Sp = S.parameters
            x = 1
            for n = 1:(length(Sp)-1)
                push!(t.args, Expr(:call, Expr(:curly, :StaticInt, (x *= Sp[n]))))
            end
            return t
        end
        if StaticArrays.SizedArray{Tuple{8,8},Float64,2,2} isa UnionAll
            @inline strides(B::StaticArrays.SizedArray{S,T,M,N,A}) where {S,T,M,N,A<:SubArray} = strides(B.data)
            parent_type(::Type{<:StaticArrays.SizedArray{S,T,M,N,A}}) where {S,T,M,N,A} = A
        else
            parent_type(::Type{<:StaticArrays.SizedArray{S,T,M,N}}) where {S,T,M,N} = Array{T,N}
        end
        @require Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e" begin
            function Adapt.adapt_storage(::Type{<:StaticArrays.SArray{S}}, xs::Array) where {S}
                StaticArrays.SArray{S}(xs)
            end
        end
    end

    @require LabelledArrays = "2ee39098-c373-598a-b85f-a56591580800" begin
        ismutable(::Type{<:LabelledArrays.LArray{T,N,Syms}}) where {T,N,Syms} = ismutable(T)
        can_setindex(::Type{<:LabelledArrays.SLArray}) = false
    end

    @require Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c" begin
        ismutable(::Type{<:Tracker.TrackedArray}) = false
        ismutable(T::Type{<:Tracker.TrackedReal}) = false
        can_setindex(::Type{<:Tracker.TrackedArray}) = false
        fast_scalar_indexing(::Type{<:Tracker.TrackedArray}) = false
        aos_to_soa(x::AbstractArray{<:Tracker.TrackedReal,N}) where {N} = Tracker.collect(x)
    end

    @require CuArrays = "3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        @require Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e" begin
            fast_scalar_indexing(::Type{<:CuArrays.CuArray}) = false
            @inline allowed_getindex(x::CuArrays.CuArray, i...) = CuArrays.@allowscalar(x[i...])
            @inline function allowed_setindex!(x::CuArrays.CuArray, v, i...)
                (CuArrays.@allowscalar(x[i...] = v))
            end

            function Base.setindex(x::CuArrays.CuArray, v, i::Int)
                _x = copy(x)
                allowed_setindex!(_x, v, i)
                return _x
            end

            function restructure(x::CuArrays.CuArray, y)
                return reshape(Adapt.adapt(parameterless_type(x), y), Base.size(x)...)
            end

            device(::Type{<:CuArrays.CuArray}) = GPU()
        end
        @require DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e" begin
            # actually do QR
            function lu_instance(A::CuArrays.CuMatrix{T}) where {T}
                return CuArrays.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
            end
        end
    end

    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        @require Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e" begin
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
        end
        @require DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e" begin
            # actually do QR
            function uu_instance(A::CUDA.CuMatrix{T}) where {T}
                return CUDA.CUSOLVER.CuQR(similar(A, 0, 0), similar(A, 0))
            end
        end
    end

    @require BandedMatrices = "aae01518-5342-5314-be14-df237901396f" begin
        struct BandedMatrixIndex <: MatrixIndex
            count::Int
            rowsize::Int
            colsize::Int
            bandinds::Array{Int,1}
            bandsizes::Array{Int,1}
            isrow::Bool
        end

        Base.firstindex(i::BandedMatrixIndex) = 1
        Base.lastindex(i::BandedMatrixIndex) = i.count
        Base.length(i::BandedMatrixIndex) = lastindex(i)
        @propagate_inbounds function Base.getindex(ind::BandedMatrixIndex, i::Int)
            @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
            _i = i
            p = 1
            while _i - ind.bandsizes[p] > 0
                _i -= ind.bandsizes[p]
                p += 1
            end
            bandind = ind.bandinds[p]
            startfromone = !xor(ind.isrow, (bandind > 0))
            if startfromone
                return _i
            else
                return _i + abs(bandind)
            end
        end

        function _bandsize(bandind, rowsize, colsize)
            -(rowsize - 1) <= bandind <= colsize - 1 || throw(ErrorException("Invalid Bandind"))
            if (bandind * (colsize - rowsize) > 0) & (abs(bandind) <= abs(colsize - rowsize))
                return min(rowsize, colsize)
            elseif bandind * (colsize - rowsize) <= 0
                return min(rowsize, colsize) - abs(bandind)
            else
                return min(rowsize, colsize) - abs(bandind) + abs(colsize - rowsize)
            end
        end

        function BandedMatrixIndex(rowsize, colsize, lowerbandwidth, upperbandwidth, isrow)
            upperbandwidth > -lowerbandwidth || throw(ErrorException("Invalid Bandwidths"))
            bandinds = upperbandwidth:-1:-lowerbandwidth
            bandsizes = [_bandsize(band, rowsize, colsize) for band in bandinds]
            BandedMatrixIndex(sum(bandsizes), rowsize, colsize, bandinds, bandsizes, isrow)
        end

        function findstructralnz(x::BandedMatrices.BandedMatrix)
            l, u = BandedMatrices.bandwidths(x)
            rowsize, colsize = Base.size(x)
            rowind = BandedMatrixIndex(rowsize, colsize, l, u, true)
            colind = BandedMatrixIndex(rowsize, colsize, l, u, false)
            return (rowind, colind)
        end

        has_sparsestruct(::Type{<:BandedMatrices.BandedMatrix}) = true
        isstructured(::Type{<:BandedMatrices.BandedMatrix}) = true
        fast_matrix_colors(::Type{<:BandedMatrices.BandedMatrix}) = true

        function matrix_colors(A::BandedMatrices.BandedMatrix)
            l, u = BandedMatrices.bandwidths(A)
            width = u + l + 1
            return _cycle(1:width, Base.size(A, 2))
        end

    end

    @require BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0" begin
        @require BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e" begin
            struct BlockBandedMatrixIndex <: MatrixIndex
                count::Int
                refinds::Array{Int,1}
                refcoords::Array{Int,1}# storing col or row inds at ref points
                isrow::Bool
            end
            Base.firstindex(i::BlockBandedMatrixIndex) = 1
            Base.lastindex(i::BlockBandedMatrixIndex) = i.count
            Base.length(i::BlockBandedMatrixIndex) = lastindex(i)
            function BlockBandedMatrixIndex(nrowblock, ncolblock, rowsizes, colsizes, l, u)
                blockrowind = BandedMatrixIndex(nrowblock, ncolblock, l, u, true)
                blockcolind = BandedMatrixIndex(nrowblock, ncolblock, l, u, false)
                sortedinds = sort(
                    [(blockrowind[i], blockcolind[i]) for i = 1:length(blockrowind)],
                    by = x -> x[1],
                )
                sort!(sortedinds, by = x -> x[2], alg = InsertionSort)# stable sort keeps the second index in order
                refinds = Array{Int,1}()
                refrowcoords = Array{Int,1}()
                refcolcoords = Array{Int,1}()
                rowheights = pushfirst!(copy(rowsizes), 1)
                cumsum!(rowheights, rowheights)
                blockheight = 0
                blockrow = 1
                blockcol = 1
                currenti = 1
                lastrowind = sortedinds[1][1] - 1
                lastcolind = sortedinds[1][2]
                for ind in sortedinds
                    rowind, colind = ind
                    if colind == lastcolind
                        if rowind > lastrowind
                            blockheight += rowsizes[rowind]
                        end
                    else
                        for j = blockcol:blockcol+colsizes[lastcolind]-1
                            push!(refinds, currenti)
                            push!(refrowcoords, blockrow)
                            push!(refcolcoords, j)
                            currenti += blockheight
                        end
                        blockcol += colsizes[lastcolind]
                        blockrow = rowheights[rowind]
                        blockheight = rowsizes[rowind]
                    end
                    lastcolind = colind
                    lastrowind = rowind
                end
                for j = blockcol:blockcol+colsizes[lastcolind]-1
                    push!(refinds, currenti)
                    push!(refrowcoords, blockrow)
                    push!(refcolcoords, j)
                    currenti += blockheight
                end
                push!(refinds, currenti)# guard
                push!(refrowcoords, -1)
                push!(refcolcoords, -1)
                rowindobj = BlockBandedMatrixIndex(currenti - 1, refinds, refrowcoords, true)
                colindobj = BlockBandedMatrixIndex(currenti - 1, refinds, refcolcoords, false)
                rowindobj, colindobj
            end
            @propagate_inbounds function Base.getindex(ind::BlockBandedMatrixIndex, i::Int)
                @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
                p = 1
                while i - ind.refinds[p] >= 0
                    p += 1
                end
                p -= 1
                _i = i - ind.refinds[p]
                if ind.isrow
                    return ind.refcoords[p] + _i
                else
                    return ind.refcoords[p]
                end
            end

            function findstructralnz(x::BlockBandedMatrices.BlockBandedMatrix)
                l, u = BlockBandedMatrices.blockbandwidths(x)
                nrowblock = BlockBandedMatrices.blocksize(x, 1)
                ncolblock = BlockBandedMatrices.blocksize(x, 2)
                rowsizes = BlockArrays.blocklengths(axes(x, 1))
                colsizes = BlockArrays.blocklengths(axes(x, 2))
                return BlockBandedMatrixIndex(
                    nrowblock,
                    ncolblock,
                    rowsizes,
                    colsizes,
                    l,
                    u,
                )
            end
            struct BandedBlockBandedMatrixIndex <: MatrixIndex
                count::Int
                refinds::Array{Int,1}
                refcoords::Array{Int,1}# storing col or row inds at ref points
                reflocalinds::Array{BandedMatrixIndex,1}
                isrow::Bool
            end
            Base.firstindex(i::BandedBlockBandedMatrixIndex) = 1
            Base.lastindex(i::BandedBlockBandedMatrixIndex) = i.count
            Base.length(i::BandedBlockBandedMatrixIndex) = lastindex(i)
            @propagate_inbounds function Base.getindex(ind::BandedBlockBandedMatrixIndex, i::Int)
                @boundscheck 1 <= i <= ind.count || throw(BoundsError(ind, i))
                p = 1
                while i - ind.refinds[p] >= 0
                    p += 1
                end
                p -= 1
                _i = i - ind.refinds[p] + 1
                ind.reflocalinds[p][_i] + ind.refcoords[p] - 1
            end


            function BandedBlockBandedMatrixIndex(
                nrowblock,
                ncolblock,
                rowsizes,
                colsizes,
                l,
                u,
                lambda,
                mu,
            )
                blockrowind = BandedMatrixIndex(nrowblock, ncolblock, l, u, true)
                blockcolind = BandedMatrixIndex(nrowblock, ncolblock, l, u, false)
                sortedinds = sort(
                    [(blockrowind[i], blockcolind[i]) for i = 1:length(blockrowind)],
                    by = x -> x[1],
                )
                sort!(sortedinds, by = x -> x[2], alg = InsertionSort)# stable sort keeps the second index in order
                rowheights = pushfirst!(copy(rowsizes), 1)
                cumsum!(rowheights, rowheights)
                colwidths = pushfirst!(copy(colsizes), 1)
                cumsum!(colwidths, colwidths)
                currenti = 1
                refinds = Array{Int,1}()
                refrowcoords = Array{Int,1}()
                refcolcoords = Array{Int,1}()
                reflocalrowinds = Array{BandedMatrixIndex,1}()
                reflocalcolinds = Array{BandedMatrixIndex,1}()
                for ind in sortedinds
                    rowind, colind = ind
                    localrowind =
                        BandedMatrixIndex(rowsizes[rowind], colsizes[colind], lambda, mu, true)
                    localcolind =
                        BandedMatrixIndex(rowsizes[rowind], colsizes[colind], lambda, mu, false)
                    push!(refinds, currenti)
                    push!(refrowcoords, rowheights[rowind])
                    push!(refcolcoords, colwidths[colind])
                    push!(reflocalrowinds, localrowind)
                    push!(reflocalcolinds, localcolind)
                    currenti += localrowind.count
                end
                push!(refinds, currenti)
                push!(refrowcoords, -1)
                push!(refcolcoords, -1)
                rowindobj = BandedBlockBandedMatrixIndex(
                    currenti - 1,
                    refinds,
                    refrowcoords,
                    reflocalrowinds,
                    true,
                )
                colindobj = BandedBlockBandedMatrixIndex(
                    currenti - 1,
                    refinds,
                    refcolcoords,
                    reflocalcolinds,
                    false,
                )
                rowindobj, colindobj
            end

            function findstructralnz(x::BlockBandedMatrices.BandedBlockBandedMatrix)
                l, u = BlockBandedMatrices.blockbandwidths(x)
                lambda, mu = BlockBandedMatrices.subblockbandwidths(x)
                nrowblock = BlockBandedMatrices.blocksize(x, 1)
                ncolblock = BlockBandedMatrices.blocksize(x, 2)
                rowsizes = BlockArrays.blocklengths(axes(x, 1))
                colsizes = BlockArrays.blocklengths(axes(x, 2))
                return BandedBlockBandedMatrixIndex(
                    nrowblock,
                    ncolblock,
                    rowsizes,
                    colsizes,
                    l,
                    u,
                    lambda,
                    mu,
                )
            end

            has_sparsestruct(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
            has_sparsestruct(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true
            isstructured(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
            isstructured(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true
            fast_matrix_colors(::Type{<:BlockBandedMatrices.BlockBandedMatrix}) = true
            fast_matrix_colors(::Type{<:BlockBandedMatrices.BandedBlockBandedMatrix}) = true

            function matrix_colors(A::BlockBandedMatrices.BlockBandedMatrix)
                l, u = BlockBandedMatrices.blockbandwidths(A)
                blockwidth = l + u + 1
                nblock = BlockBandedMatrices.blocksize(A, 2)
                cols = BlockArrays.blocklengths(axes(A, 2))
                blockcolors = _cycle(1:blockwidth, nblock)
                # the reserved number of colors of a block is the maximum length of columns of blocks with the same block color
                ncolors = [maximum(cols[i:blockwidth:nblock]) for i = 1:blockwidth]
                endinds = cumsum(ncolors)
                startinds = [endinds[i] - ncolors[i] + 1 for i = 1:blockwidth]
                colors = [
                    (startinds[blockcolors[i]]:endinds[blockcolors[i]])[1:cols[i]]
                    for i = 1:nblock
                ]
                return reduce(vcat,colors)
            end

            function matrix_colors(A::BlockBandedMatrices.BandedBlockBandedMatrix)
                l, u = BlockBandedMatrices.blockbandwidths(A)
                lambda, mu = BlockBandedMatrices.subblockbandwidths(A)
                blockwidth = l + u + 1
                subblockwidth = lambda + mu + 1
                nblock = BlockBandedMatrices.blocksize(A, 2)
                cols = BlockArrays.blocklengths(axes(A, 2))
                blockcolors = _cycle(1:blockwidth, nblock)
                # the reserved number of colors of a block is the min of subblockwidth and the largest length of columns of blocks with the same block color
                ncolors = [
                    min(subblockwidth, maximum(cols[i:blockwidth:nblock]))
                    for i = 1:min(blockwidth, nblock)
                ]
                endinds = cumsum(ncolors)
                startinds = [endinds[i] - ncolors[i] + 1 for i = 1:min(blockwidth, nblock)]
                colors = [
                    _cycle(startinds[blockcolors[i]]:endinds[blockcolors[i]], cols[i])
                    for i = 1:nblock
                ]
                return reduce(vcat,colors)
            end
        end
    end
    @require OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881" begin
        relative_offsets(r::OffsetArrays.IdOffsetRange) = (getfield(r, :offset),)
        relative_offsets(A::OffsetArrays.OffsetArray) = getfield(A, :offsets)
        function relative_offsets(A::OffsetArrays.OffsetArray, ::StaticInt{dim}) where {dim}
            if dim > ndims(A)
                return static(0)
            else
                return getfield(relative_offsets(A), dim)
            end
        end
        function relative_offsets(A::OffsetArrays.OffsetArray, dim::Int)
            if dim > ndims(A)
                return 0
            else
                return getfield(relative_offsets(A), dim)
            end
        end
        ArrayInterface.parent_type(::Type{<:OffsetArrays.OffsetArray{T,N,A}}) where {T,N,A} = A
        function _offset_axis_type(::Type{T}, dim::StaticInt{D}) where {T,D}
            OffsetArrays.IdOffsetRange{Int,ArrayInterface.axes_types(T, dim)}
        end
        function ArrayInterface.axes_types(::Type{T}) where {T<:OffsetArrays.OffsetArray}
            Static.eachop_tuple(_offset_axis_type, Static.nstatic(Val(ndims(T))), ArrayInterface.parent_type(T))
        end
        function ArrayInterface.known_offsets(::Type{A}) where {A<:OffsetArrays.OffsetArray}
            ntuple(identity -> nothing, Val(ndims(A)))
        end
        function ArrayInterface.offsets(A::OffsetArrays.OffsetArray)
            map(+, ArrayInterface.offsets(parent(A)), relative_offsets(A))
        end
       @inline function ArrayInterface.offsets(A::OffsetArrays.OffsetArray, dim)
            d = ArrayInterface.to_dims(A, dim)
            ArrayInterface.offsets(parent(A), d) + relative_offsets(A, d)
        end
        @inline function ArrayInterface.axes(A::OffsetArrays.OffsetArray)
            map(OffsetArrays.IdOffsetRange, ArrayInterface.axes(parent(A)), relative_offsets(A))
        end
        @inline function ArrayInterface.axes(A::OffsetArrays.OffsetArray, dim)
            d = to_dims(A, dim)
            OffsetArrays.IdOffsetRange(ArrayInterface.axes(parent(A), d), relative_offsets(A, d))
        end
    end
end

end
