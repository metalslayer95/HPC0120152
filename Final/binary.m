function [output] = binary(input)
[r,c] = size(input);
output = zeros(r,c);
one = find(input > 0);
output(one) = 1;