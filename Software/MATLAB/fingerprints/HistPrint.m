% Master's thesis
% File description: Produces a histogram print for one image
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

function [output] = HistPrint(input, bins)
    [a, b, ch] = size(input);
    output = zeros(ch, bins);
    for i=1:ch
        image = input(:,:,i);
        histPiece = histcounts(image, bins);
        output(i,:) = histPiece;
    end

	output = min(output, 255)/255;
end
