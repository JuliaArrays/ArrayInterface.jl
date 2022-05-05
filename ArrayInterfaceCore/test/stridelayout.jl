@testset "offsets" begin
    @test @inferred(ArrayInterfaceCore.known_offsets(A)) === (1, 1, 1)
    @test @inferred(ArrayInterfaceCore.known_offsets(Ap)) === (1, 1)
    @test @inferred(ArrayInterfaceCore.known_offsets(Ar)) === (1, 1, 1)
    @test @inferred(ArrayInterfaceCore.known_offsets(Ar, static(1))) === 1
    @test @inferred(ArrayInterfaceCore.known_offsets(Ar, static(4))) === 1
    @test @inferred(ArrayInterfaceCore.known_offsets(A2)) === (1, 1, 1)
    @test @inferred(ArrayInterfaceCore.known_offsets(A2r)) === (1, 1, 1)
    @test @inferred(ArrayInterfaceCore.offsets(A)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterfaceCore.offsets(Ap)) === (StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterfaceCore.offsets(Ar)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterfaceCore.offsets(A2)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterfaceCore.offsets(A2r)) === (StaticInt(1), StaticInt(1), StaticInt(1))

    @test @inferred(ArrayInterfaceCore.offsets(S)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterfaceCore.offsets(Sp)) === (StaticInt(1), StaticInt(1), StaticInt(1))
    @test @inferred(ArrayInterfaceCore.offsets(Sp2)) === (StaticInt(1), StaticInt(1), StaticInt(1))

    @test @inferred(ArrayInterfaceCore.known_offsets(S)) === (1, 1, 1)
    @test @inferred(ArrayInterfaceCore.known_offsets(Sp)) === (1, 1, 1)
    @test @inferred(ArrayInterfaceCore.known_offsets(Sp2)) === (1, 1, 1)

    @test @inferred(ArrayInterfaceCore.known_offsets(R)) === (1,)
    @test @inferred(ArrayInterfaceCore.known_offsets(Rnr)) === (1,)
    @test @inferred(ArrayInterfaceCore.known_offsets(1:10)) === (1,)
end

@testset "strides" begin
    @test @inferred(ArrayInterfaceCore.strides(A)) === (StaticInt(1), 3, 12)
    @test @inferred(ArrayInterfaceCore.strides(Ap)) === (StaticInt(1), 12)
    @test @inferred(ArrayInterfaceCore.strides(A)) == strides(A)
    @test @inferred(ArrayInterfaceCore.strides(Ap)) == strides(Ap)
    @test @inferred(ArrayInterfaceCore.strides(Ar)) === (StaticInt{1}(), 6, 24)
    @test @inferred(ArrayInterfaceCore.strides(A2)) === (StaticInt(1), 4, 12)
    @test @inferred(ArrayInterfaceCore.strides(A2r)) === (StaticInt(1), 2, 6)

    @test @inferred(ArrayInterfaceCore.strides(S)) === (StaticInt(1), StaticInt(2), StaticInt(6))
    @test @inferred(ArrayInterfaceCore.strides(Sp)) === (StaticInt(6), StaticInt(1), StaticInt(2))
    @test @inferred(ArrayInterfaceCore.strides(Sp2)) === (StaticInt(6), StaticInt(2), StaticInt(1))
    @test @inferred(ArrayInterfaceCore.strides(view(Sp2, :, 1, 1)')) === (StaticInt(6), StaticInt(6))

    @test @inferred(ArrayInterfaceCore.stride(Sp2, StaticInt(1))) === StaticInt(6)
    @test @inferred(ArrayInterfaceCore.stride(Sp2, StaticInt(2))) === StaticInt(2)
    @test @inferred(ArrayInterfaceCore.stride(Sp2, StaticInt(3))) === StaticInt(1)
    @test @inferred(ArrayInterfaceCore.strides(Mp)) === (StaticInt(2), StaticInt(6))
    @test @inferred(ArrayInterfaceCore.strides(Mp2)) === (StaticInt(1), StaticInt(6))
    @test @inferred(ArrayInterfaceCore.strides(Mp)) == strides(Mp)
    @test @inferred(ArrayInterfaceCore.strides(Mp2)) == strides(Mp2)

    @test_throws MethodError ArrayInterfaceCore.strides(DummyZeros(3,4))

    @test @inferred(ArrayInterfaceCore.known_strides(A)) === (1, nothing, nothing)
    @test @inferred(ArrayInterfaceCore.known_strides(Ap)) === (1, nothing)
    @test @inferred(ArrayInterfaceCore.known_strides(Ar)) === (1, nothing, nothing)
    @test @inferred(ArrayInterfaceCore.known_strides(reshape(view(zeros(100), 1:60), (3,4,5)))) === (1, nothing, nothing)
    @test @inferred(ArrayInterfaceCore.known_strides(A2)) === (1, nothing, nothing)
    @test @inferred(ArrayInterfaceCore.known_strides(A2r)) === (1, nothing, nothing)

    @test @inferred(ArrayInterfaceCore.known_strides(S)) === (1, 2, 6)
    @test @inferred(ArrayInterfaceCore.known_strides(Sp)) === (6, 1, 2)
    @test @inferred(ArrayInterfaceCore.known_strides(Sp2)) === (6, 2, 1)
    @test @inferred(ArrayInterfaceCore.known_strides(Sp2, StaticInt(1))) === 6
    @test @inferred(ArrayInterfaceCore.known_strides(Sp2, StaticInt(2))) === 2
    @test @inferred(ArrayInterfaceCore.known_strides(Sp2, StaticInt(3))) === 1
    @test @inferred(ArrayInterfaceCore.known_strides(Sp2, StaticInt(4))) === ArrayInterfaceCore.known_length(Sp2)
    @test @inferred(ArrayInterfaceCore.known_strides(view(Sp2, :, 1, 1)')) === (6, 6)
end

@testset "Static-Dynamic Size, Strides, and Offsets" begin
    if VERSION ≥ v"1.6.0-DEV.1581"
        colors = [(R = rand(), G = rand(), B = rand()) for i ∈ 1:100];

        colormat = reinterpret(reshape, Float64, colors)
        @test @inferred(ArrayInterfaceCore.strides(colormat)) === (StaticInt(1), StaticInt(3))
        @test @inferred(ArrayInterfaceCore.dense_dims(colormat)) === (True(),True())
        @test @inferred(ArrayInterfaceCore.dense_dims(view(colormat,:,4))) === (True(),)
        @test @inferred(ArrayInterfaceCore.dense_dims(view(colormat,:,4:7))) === (True(),True())
        @test @inferred(ArrayInterfaceCore.dense_dims(view(colormat,2:3,:))) === (True(),False())

        Rr = reinterpret(reshape, Int32, R)
        @test @inferred(ArrayInterfaceCore.size(Rr)) === (StaticInt(2),StaticInt(2))
        @test @inferred(ArrayInterfaceCore.known_size(Rr)) === (2, 2)

        Sr = Wrapper(reinterpret(reshape, Complex{Int64}, S))
        @test @inferred(ArrayInterfaceCore.size(Sr)) == (static(3), static(4))
        @test @inferred(ArrayInterfaceCore.known_size(Sr)) === (3, 4)
        @test @inferred(ArrayInterfaceCore.strides(Sr)) === (static(1), static(3))
        Sr2 = reinterpret(reshape, Complex{Int64}, S);
        @test @inferred(ArrayInterfaceCore.dense_dims(Sr2)) === (True(),True())
        @test @inferred(ArrayInterfaceCore.dense_dims(view(Sr2,:,2))) === (True(),)
        @test @inferred(ArrayInterfaceCore.dense_dims(view(Sr2,:,2:3))) === (True(),True())
        @test @inferred(ArrayInterfaceCore.dense_dims(view(Sr2,2:3,:))) === (True(),False())

        Ar2c = reinterpret(reshape, Complex{Float64}, view(rand(2, 5, 7), :, 2:4, 3:5));
        @test @inferred(ArrayInterfaceCore.strides(Ar2c)) === (StaticInt(1), 5)
        Ar2c_static = reinterpret(reshape, Complex{Float64}, view(@MArray(rand(2, 5, 7)), :, 2:4, 3:5));
        @test @inferred(ArrayInterfaceCore.strides(Ar2c_static)) === (StaticInt(1), StaticInt(5))

        Ac2r = reinterpret(reshape, Float64, view(rand(ComplexF64, 5, 7), 2:4, 3:6));
        @test @inferred(ArrayInterfaceCore.strides(Ac2r)) === (StaticInt(1), StaticInt(2), 10)
        Ac2r_static = reinterpret(reshape, Float64, view(@MMatrix(rand(ComplexF64, 5, 7)), 2:4, 3:6));
        @test @inferred(ArrayInterfaceCore.strides(Ac2r_static)) === (StaticInt(1), StaticInt(2), StaticInt(10))

        Ac2t = reinterpret(reshape, Tuple{Float64,Float64}, view(rand(ComplexF64, 5, 7), 2:4, 3:6));
        @test @inferred(ArrayInterfaceCore.strides(Ac2t)) === (StaticInt(1), 5)
        Ac2t_static = reinterpret(reshape, Tuple{Float64,Float64}, view(@MMatrix(rand(ComplexF64, 5, 7)), 2:4, 3:6));
        @test @inferred(ArrayInterfaceCore.strides(Ac2t_static)) === (StaticInt(1), StaticInt(5))
    end
end

@testset "Memory Layout" begin
    x = zeros(100);
    @test ArrayInterfaceCore.size(Base.Broadcast.Broadcasted(+, (x, x'))) === (100,100)
    # R = reshape(view(x, 1:100), (10,10));
    # A = zeros(3,4,5);
    A = Wrapper(reshape(view(x, 1:60), (3,4,5)));
    B = A .== 0;
    D1 = view(A, 1:2:3, :, :);  # first dimension is discontiguous
    D2 = view(A, :, 2:2:4, :);  # first dimension is contiguous

    @test @inferred(ArrayInterfaceCore.defines_strides(x))
    @test @inferred(ArrayInterfaceCore.defines_strides(A))
    @test @inferred(ArrayInterfaceCore.defines_strides(D1))
    @test !@inferred(ArrayInterfaceCore.defines_strides(view(A, :, [1,2],1)))
    @test @inferred(ArrayInterfaceCore.defines_strides(DenseWrapper{Int,2,Matrix{Int}}))

    @test @inferred(device(A)) === ArrayInterfaceCore.CPUPointer()
    @test @inferred(device(B)) === ArrayInterfaceCore.CPUIndex()
    @test @inferred(device(-1:19)) === ArrayInterfaceCore.CPUIndex()
    @test @inferred(device((1,2,3))) === ArrayInterfaceCore.CPUTuple()
    @test @inferred(device(PermutedDimsArray(A,(3,1,2)))) === ArrayInterfaceCore.CPUPointer()
    @test @inferred(device(view(A, 1, :, 2:4))) === ArrayInterfaceCore.CPUPointer()
    @test @inferred(device(view(A, 1, :, 2:4)')) === ArrayInterfaceCore.CPUPointer()
    @test isnothing(device("Hello, world!"))
    @test @inferred(device(DenseWrapper{Int,2,Matrix{Int}})) === ArrayInterfaceCore.CPUPointer()
    #=
    @btime ArrayInterfaceCore.contiguous_axis($(reshape(view(zeros(100), 1:60), (3,4,5))))
      0.047 ns (0 allocations: 0 bytes)
    =#
    @test @inferred(contiguous_axis(A)) === ArrayInterfaceCore.StaticInt(1)
    @test @inferred(contiguous_axis(B)) === ArrayInterfaceCore.StaticInt(1)
    @test @inferred(contiguous_axis(-1:19)) === ArrayInterfaceCore.StaticInt(1)
    @test @inferred(contiguous_axis(D1)) === ArrayInterfaceCore.StaticInt(-1)
    @test @inferred(contiguous_axis(D2)) === ArrayInterfaceCore.StaticInt(1)
    @test @inferred(contiguous_axis(PermutedDimsArray(A,(3,1,2)))) === ArrayInterfaceCore.StaticInt(2)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === ArrayInterfaceCore.StaticInt(1)
    @test @inferred(contiguous_axis(transpose(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])))) === ArrayInterfaceCore.StaticInt(2)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === ArrayInterfaceCore.StaticInt(2)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === ArrayInterfaceCore.StaticInt(-1)
    @test @inferred(contiguous_axis(PermutedDimsArray(@view(A[2,:,:]),(2,1)))) === ArrayInterfaceCore.StaticInt(-1)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === ArrayInterfaceCore.StaticInt(-1)
    @test @inferred(contiguous_axis(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === ArrayInterfaceCore.StaticInt(1)
    @test @inferred(contiguous_axis((3,4))) === StaticInt(1)
    @test @inferred(contiguous_axis(rand(4)')) === StaticInt(2)
    @test @inferred(contiguous_axis(view(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])', :, 1)')) === StaticInt(-1)
    @test @inferred(contiguous_axis(DummyZeros(3,4))) === nothing
    @test @inferred(contiguous_axis(PermutedDimsArray(DummyZeros(3,4), (2, 1)))) === nothing
    @test @inferred(contiguous_axis(view(DummyZeros(3,4), 1, :))) === nothing
    @test @inferred(contiguous_axis(view(DummyZeros(3,4), 1, :)')) === nothing


    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(A)) == (true,false,false)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(B)) == (true,false,false)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(-1:10)) == (true,)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(PermutedDimsArray(A,(3,1,2)))) == (false,true,false)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) == (true,false)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) == (false,true)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) == (false,true,false)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) == (false,false)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) == (false,false)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) == (true,false)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,[1,3,4]]))) == (false,true,false)
    @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(DummyZeros(3,4))) === nothing

    @test @inferred(contiguous_batch_size(A)) === ArrayInterfaceCore.StaticInt(0)
    @test @inferred(contiguous_batch_size(B)) === ArrayInterfaceCore.StaticInt(0)
    @test @inferred(contiguous_batch_size(-1:18)) === ArrayInterfaceCore.StaticInt(0)
    @test @inferred(contiguous_batch_size(PermutedDimsArray(A,(3,1,2)))) === ArrayInterfaceCore.StaticInt(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === ArrayInterfaceCore.StaticInt(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) === ArrayInterfaceCore.StaticInt(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === ArrayInterfaceCore.StaticInt(0)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === ArrayInterfaceCore.StaticInt(-1)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === ArrayInterfaceCore.StaticInt(-1)
    @test @inferred(contiguous_batch_size(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === ArrayInterfaceCore.StaticInt(0)
    let u_base = randn(10, 10)
        u_view = view(u_base, 3, :)
        u_reshaped_view = reshape(u_view, 1, size(u_base, 2))
        @test @inferred(contiguous_batch_size(u_view)) === ArrayInterfaceCore.StaticInt(-1)
        @test @inferred(contiguous_batch_size(u_reshaped_view)) === ArrayInterfaceCore.StaticInt(-1)
    end

    @test @inferred(stride_rank(A)) == (1,2,3)
    @test @inferred(stride_rank(B)) == (1,2,3)
    @test @inferred(stride_rank(-4:4)) == (1,)
    @test @inferred(stride_rank(view(A,:,:,1))) === (static(1), static(2))
    @test @inferred(stride_rank(view(A,:,:,1))) === ((ArrayInterfaceCore.StaticInt(1),ArrayInterfaceCore.StaticInt(2)))
    @test @inferred(stride_rank(PermutedDimsArray(A,(3,1,2)))) == (3, 1, 2)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) == (1, 2)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) == (2, 1)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) == (3, 1, 2)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) == (3, 2)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) == (2, 3)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) == (1, 3)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[:,2,1])')) == (2, 1)
    @test @inferred(stride_rank(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,[1,3,4]]))) == (3, 1, 2)
    @test @inferred(stride_rank(DummyZeros(3,4)')) === nothing
    @test @inferred(stride_rank(PermutedDimsArray(DummyZeros(3,4), (2, 1)))) === nothing
    @test @inferred(stride_rank(view(DummyZeros(3,4), 1, :))) === nothing
    uA = reinterpret(reshape, UInt64, A)
    @test @inferred(stride_rank(uA)) === stride_rank(A)
    rA = reinterpret(reshape, Tuple{3,Float64}, A)
    @test @inferred(stride_rank(rA)) === (static(1), static(2))
    cA = copy(rA)
    rcA = reinterpret(reshape, Float64, cA)
    @test @inferred(stride_rank(rcA)) === stride_rank(A)

    #=
    @btime ArrayInterfaceCore.is_column_major($(PermutedDimsArray(A,(3,1,2))))
      0.047 ns (0 allocations: 0 bytes)
    @btime ArrayInterfaceCore.is_column_major($(ArrayInterfaceCore.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))))
      0.047 ns (0 allocations: 0 bytes)
    @btime ArrayInterfaceCore.is_column_major($(ArrayInterfaceCore.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')))
      0.047 ns (0 allocations: 0 bytes)

      PermutedDimsArray(A,(3,1,2))[2:3,1:2,:])
      @view(PermutedDimsArray(reshape(view(zeros(100), 1:60), (3,4,5)), (3,1,2)), 2:3, 1:2, :)
    =#

    @test @inferred(ArrayInterfaceCore.is_column_major(A)) === True()
    @test @inferred(ArrayInterfaceCore.is_column_major(B)) === True()
    @test @inferred(ArrayInterfaceCore.is_column_major(-4:7)) === False()
    @test @inferred(ArrayInterfaceCore.is_column_major(PermutedDimsArray(A,(3,1,2)))) === False()
    @test @inferred(ArrayInterfaceCore.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) === True()
    @test @inferred(ArrayInterfaceCore.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) === False()
    @test @inferred(ArrayInterfaceCore.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === False()
    @test @inferred(ArrayInterfaceCore.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) === False()
    @test @inferred(ArrayInterfaceCore.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) === True()
    @test @inferred(ArrayInterfaceCore.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) === True()
    @test @inferred(ArrayInterfaceCore.is_column_major(@view(PermutedDimsArray(A,(3,1,2))[:,2,1])')) === False()
    @test @inferred(ArrayInterfaceCore.is_column_major(2.3)) === False()

    @test @inferred(dense_dims(A)) == (true,true,true)
    @test @inferred(dense_dims(B)) == (true,true,true)
    @test @inferred(dense_dims(-3:9)) == (true,)
    @test @inferred(dense_dims(PermutedDimsArray(A,(3,1,2)))) == (true,true,true)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:]))) == (true,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2,1:2,:])')) == (false,true)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) == (false,true,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,:,1:2]))) == (false,true,true)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:]))) == (false,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,2,:])')) == (false,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[:,1:2,1])')) == (true,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,:,[1,2]]))) == (false,true,false)
    @test @inferred(dense_dims(@view(PermutedDimsArray(A,(3,1,2))[2:3,[1,2,3],:]))) == (false,false,false)
    # TODO Currently Wrapper can't function the same as Array because Array can change
    # the dimensions on reshape. We should be rewrapping the result in `Wrapper` but we
    # first need to develop a standard method for reconstructing arrays
    @test @inferred(dense_dims(vec(parent(A)))) == (true,)
    @test @inferred(dense_dims(vec(parent(A))')) == (true,true)
    @test @inferred(dense_dims(DummyZeros(3,4))) === nothing
    @test @inferred(dense_dims(DummyZeros(3,4)')) === nothing
    @test @inferred(dense_dims(PermutedDimsArray(DummyZeros(3,4), (2, 1)))) === nothing
    @test @inferred(dense_dims(view(DummyZeros(3,4), :, 1))) === nothing
    @test @inferred(dense_dims(view(DummyZeros(3,4), :, 1)')) === nothing
    @test @inferred(ArrayInterfaceCore.is_dense(A)) === @inferred(ArrayInterfaceCore.is_dense(A)) === @inferred(ArrayInterface.is_dense(PermutedDimsArray(A,(3,1,2)))) === @inferred(ArrayInterface.is_dense(Array{Float64,0}(undef))) === True()
    @test @inferred(ArrayInterfaceCore.is_dense(@view(PermutedDimsArray(A,(3,1,2))[2:3,1:2,:]))) === @inferred(ArrayInterfaceCore.is_dense(@view(PermutedDimsArray(A,(3,1,2))[2:3,:,[1,2]]))) === @inferred(ArrayInterface.is_dense(@view(PermutedDimsArray(A,(3,1,2))[2:3,[1,2,3],:]))) === False()
  
    C = Array{Int8}(undef, 2,2,2,2);
    doubleperm = PermutedDimsArray(PermutedDimsArray(C,(4,2,3,1)), (4,2,1,3));
    @test collect(strides(C))[collect(stride_rank(doubleperm))] == collect(strides(doubleperm))

    if isdefined(Base, :ReshapedReinterpretArray) # reinterpret(reshape,...) tests
        C1 = reinterpret(reshape, Float64, PermutedDimsArray(Array{Complex{Float64}}(undef, 3,4,5), (2,1,3)));
        C2 = reinterpret(reshape, Complex{Float64}, PermutedDimsArray(view(A,1:2,:,:), (1,3,2)));
        C3 = reinterpret(reshape, Complex{Float64}, PermutedDimsArray(Wrapper(reshape(view(x, 1:24), (2,3,4))), (1,3,2)));

        @test @inferred(ArrayInterfaceCore.defines_strides(C1))
        @test @inferred(ArrayInterfaceCore.defines_strides(C2))
        @test @inferred(ArrayInterfaceCore.defines_strides(C3))

        @test @inferred(device(C1)) === ArrayInterfaceCore.CPUPointer()
        @test @inferred(device(C2)) === ArrayInterfaceCore.CPUPointer()
        @test @inferred(device(C3)) === ArrayInterfaceCore.CPUPointer()

        @test @inferred(contiguous_batch_size(C1)) === ArrayInterfaceCore.StaticInt(0)
        @test @inferred(contiguous_batch_size(C2)) === ArrayInterfaceCore.StaticInt(0)
        @test @inferred(contiguous_batch_size(C3)) === ArrayInterfaceCore.StaticInt(0)

        @test @inferred(stride_rank(C1)) == (1,3,2,4)
        @test @inferred(stride_rank(C2)) == (2,1)
        @test @inferred(stride_rank(C3)) == (2,1)

        @test @inferred(contiguous_axis(C1)) === StaticInt(1)
        @test @inferred(contiguous_axis(C2)) === StaticInt(0)
        @test @inferred(contiguous_axis(C3)) === StaticInt(2)

        @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(C1)) == (true,false,false,false)
        @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(C2)) == (false,false)
        @test @inferred(ArrayInterfaceCore.contiguous_axis_indicator(C3)) == (false,true)

        @test @inferred(ArrayInterfaceCore.is_column_major(C1)) === False()
        @test @inferred(ArrayInterfaceCore.is_column_major(C2)) === False()
        @test @inferred(ArrayInterfaceCore.is_column_major(C3)) === False()

        @test @inferred(dense_dims(C1)) == (true,true,true,true)
        @test @inferred(dense_dims(C2)) == (false,false)
        @test @inferred(dense_dims(C3)) == (true,true)
    end
end

