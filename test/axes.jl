
@test @inferred(ArrayInterface.axes(Array{Float64}(undef, 4)')) === (StaticInt(1):StaticInt(1),Base.OneTo(4))
@test @inferred(ArrayInterface.axes(Array{Float64}(undef, 4, 3)')) === (Base.OneTo(3),Base.OneTo(4))

@testset "LazyAxis" begin
    A = zeros(3,4,5);
    SA = @SArray(zeros(3,4,5))
    lz1 = ArrayInterface.LazyAxis{1}(A)
    slz1 = ArrayInterface.LazyAxis{1}(SA)
    lzc = ArrayInterface.LazyAxis{:}(A)
    slzc = ArrayInterface.LazyAxis{:}(SA)

    @test @inferred(first(lz1)) === @inferred(first(slz1))
    @test @inferred(first(lzc)) === @inferred(first(slzc))
    @test @inferred(last(lz1)) === @inferred(last(slz1))
    @test @inferred(last(lzc)) === @inferred(last(slzc))
    @test @inferred(length(lz1)) === @inferred(length(slz1))
    @test @inferred(length(lzc)) === @inferred(length(slzc))
    @test @inferred(Base.to_shape(lzc)) == length(slzc)
    @test @inferred(Base.checkindex(Bool, lzc, 1)) & @inferred(Base.checkindex(Bool, slzc, 1))

    @test @inferred(ArrayInterface.parent_type(ArrayInterface.LazyAxis{4}(A))) <: ArrayInterface.SOneTo{1}
    @test ArrayInterface.can_change_size(ArrayInterface.LazyAxis{1,Vector{Any}})

    Aperm = PermutedDimsArray(A, (3,1,2))
    Aview = @view(Aperm[:,1:2,1])
    @test map(parent, @inferred(ArrayInterface.lazy_axes(Aperm))) === @inferred(ArrayInterface.axes(Aperm))
    @test map(parent, @inferred(ArrayInterface.lazy_axes(Aview))) === @inferred(ArrayInterface.axes(Aview))
    @test map(parent, @inferred(ArrayInterface.lazy_axes(Aview'))) === @inferred(ArrayInterface.axes(Aview'))
    @test map(parent, @inferred(ArrayInterface.lazy_axes((1:2)'))) === @inferred(ArrayInterface.axes((1:2)'))
end

if isdefined(Base, :ReshapedReinterpretArray)
    a = rand(3, 5)
    ua = reinterpret(reshape, UInt64, a)
    @test ArrayInterface.axes(ua) === ArrayInterface.axes(a)
    @test ArrayInterface.axes(ua, 1) === ArrayInterface.axes(a, 1)
    @test @inferred(ArrayInterface.axes(ua)) isa ArrayInterface.axes_types(ua)
    u8a = reinterpret(reshape, UInt8, a)
    @test @inferred(ArrayInterface.axes(u8a)) isa ArrayInterface.axes_types(u8a)
    @test @inferred(ArrayInterface.axes(u8a, static(1))) isa ArrayInterface.axes_types(u8a, 1)
    @test @inferred(ArrayInterface.axes(u8a, static(2))) isa ArrayInterface.axes_types(u8a, 2)
    fa = reinterpret(reshape, Float64, copy(u8a))
    @inferred(ArrayInterface.axes(fa)) isa ArrayInterface.axes_types(fa)
end
