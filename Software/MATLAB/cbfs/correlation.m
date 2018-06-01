% Master's thesis
% File description: Finds the correlation matrix of a single
% 					multispectral image
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

function [theta_matrix] = correlation(img)
	[rows, cols, channels] = size(img);
	img_vec = double(reshape(img, rows*cols, channels));
	theta_matrix = zeros(rows, cols);

	for i=1:channels
		for j=i:channels
			theta = dot(img_vec(:,i), img_vec(:,j))/(norm(img_vec(:,i)) * norm(img_vec(:,j)));
			theta_matrix(i,j) = theta;
		end
	end
end