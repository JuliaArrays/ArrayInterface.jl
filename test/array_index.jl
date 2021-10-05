
function test_layout(x)
    @testset "$x" begin
        linbuf, linlyt = @inferred(ArrayInterface.layout(x, static(1)))
        for i in eachindex(IndexLinear(), x)
            @test @inferred(linbuf[linlyt[i]]) == x[i]
        end
        carbuf, carlyt = @inferred(ArrayInterface.layout(x, static(ndims(x))))
        for i in eachindex(IndexCartesian(), x)
            @test @inferred(carbuf[carlyt[i]]) == x[i]
        end
    end
    return nothing
end

A = rand(4,4,4);
A[:] .= eachindex(A);
Aperm = PermutedDimsArray(A,(3,1,2));
Aview = @view(Aperm[:,1:2,1]);
Ap = Aview';
Apperm = PermutedDimsArray(Ap, (2, 1));

test_layout(A)
test_layout(Aperm)
test_layout(Aview)
test_layout(Ap)
test_layout(Apperm)

