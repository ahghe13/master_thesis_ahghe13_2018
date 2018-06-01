% Master's thesis
% File description: Draws a white rectangle according to the input
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

function [output] = draw_rect(input, x_min, y_min, width, height)
	output = input;
	output(y_min:y_min,x_min:x_min+width) = 2^8;
	output(y_min+height:y_min+height,x_min:x_min+width) = 2^8;
	output(y_min:y_min+height,x_min:x_min) = 2^8;
	output(y_min:y_min+height,x_min+width:x_min+width) = 2^8;
end