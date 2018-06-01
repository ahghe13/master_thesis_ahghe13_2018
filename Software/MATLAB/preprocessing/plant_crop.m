% Master's thesis
% File description: Crops a single multispectral image
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

function [output, crop] = plant_crop(input, ref_channel)
	[x_min y_min, width, height] = Plant_boundaries(input(:,:,ref_channel));

	output = zeros(height+1, width+1, size(input, 3));

	size(output)
	for i=1:size(input,3)
		output(:,:,i) = imcrop(input(:,:,i),[x_min y_min width height]);
	end

	crop = zeros(size(input,1), size(input,2));
	crop = input(:,:,ref_channel);
	crop(y_min:y_min,x_min:x_min+width) = 2^8;
	crop(y_min+height:y_min+height,x_min:x_min+width) = 2^8;
	crop(y_min:y_min+height,x_min:x_min) = 2^8;
	crop(y_min:y_min+height,x_min+width:x_min+width) = 2^8;
end