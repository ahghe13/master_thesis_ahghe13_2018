% Master's thesis
% File description: Slices a single image
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

function [output, slices] = image_slicer(input, dim)
	[height, width, channels] = size(input);

	h_slices = floor(height/dim);
	h_offset = 1+floor((height/dim-h_slices)*dim/2);
	w_slices = floor(width/dim);
	w_offset = 1+floor((width/dim-w_slices)*dim/2);

	output_size = h_slices*w_slices;

	slices = input(:,:,164);
	output = zeros(dim, dim, channels, output_size);

	dim = dim-1;
	counter = 1;
	for i=1:h_slices
		for j=1:w_slices
			x_min = dim*(j-1)+w_offset;
			y_min = dim*(i-1)+h_offset;
			for k=1:channels
				output(:,:,k,counter) = imcrop(input(:,:,k),[x_min y_min dim dim]);
			end
			slices = draw_rect(slices, x_min, y_min, dim, dim);
			counter = counter + 1;
		end
	end

end