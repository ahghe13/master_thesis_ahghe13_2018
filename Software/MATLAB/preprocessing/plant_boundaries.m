% Master's thesis
% File description: Finds the boundaries of the plant
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

function [x_min,y_min,width,height] = plant_boundaries(input)
	[a, b, ch] = size(input);

	x_min = floor(b/2);
	y_min = floor(a/2);
	x_max = floor(0.5+b/2);
	y_max = floor(0.5+a/2);

	for i=1:a
		for j=1:b
			if input(i,j) > (0.4*2^8)
				if x_min > j; x_min = j; end
				if y_min > i; y_min = i; end
				if x_max < j; x_max = j; end
				if y_max < i; y_max = i; end
			end
		end
	end

	width = x_max-x_min;
	height = y_max-y_min;
end



