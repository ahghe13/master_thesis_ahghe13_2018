% Master's thesis
% File description: Finds the correlation matrix of the .tif
% 					files in the given directory paths
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

clear;
addpath('../tiff_handling');

no_of_channels = 10;

tif_data_path = [];		% One or more comma separated paths to data
save_path = ''			% Path to save the correlation matrix (must be a .csv file)

theta = zeros(224,224);
n = 0;

for i=1:size(tif_data_path,1)
	data_path = tif_data_path(i,:)

	l = dir(strcat(data_path, '*.tif'));

	[files, c] = size(l);

	fprintf(['\nProcessing images in ', data_path, '\n'])

	for j=1:2
	    tif_name = l(j).name;
	    
	    img = loadtiff(strcat(data_path, tif_name));
	    theta = theta + correlation(img);
	    n = n + 1;

	    fprintf(['Image ', num2str(j), ' finished!\n'])
	end
end
theta = theta/n;
csvwrite(save_path, theta)
