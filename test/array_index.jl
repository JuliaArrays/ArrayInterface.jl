
function test_layout(x)
    @testset "$x" begin
        linear_lyt = ArrayInterface.instantiate(ArrayInterface.layout(x, ArrayInterface.AccessElement{1}()))
        for i in eachindex(IndexLinear(), x)
            @test linear_lyt[i] == x[i]
        end
        cartesian_lyt = ArrayInterface.instantiate(ArrayInterface.layout(x, ArrayInterface.AccessElement{ndims(x)}()))
        for i in eachindex(IndexCartesian(), x)
            @test cartesian_lyt[i] == x[i]
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

