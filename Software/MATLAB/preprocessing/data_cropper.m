% Master's thesis
% File description: Crops the data
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

clear;
addpath('../tiff_handling');

root_path = ''		% Root path to data

input_dirs = [];	% One or more paths to data
output_dirs = [];	% Similar number of paths to save the output

for i=1:size(input_dirs,1)
	data_path = strcat(root_path, input_dirs(i,:));

	l = dir(strcat(data_path, '*.tif'));

	[files, c] = size(l);

	output_path = strcat(root_path, output_dirs(i,:));
	mkdir(output_path);

	validation_im_path = strcat(output_path, '/crop_validation');
	mkdir(validation_im_path);

	fprintf(['\nProcessing images in ', output_path, '\n']);

	for j=1:files
	    tif_name = l(j).name;
	    
	    tif = imread(strcat(data_path, tif_name));
	    tif = bitshift(tif, -4);
	    tif = uint8(tif);

	    [out, crop] = Plant_crop(tif, 164);

	    clear options;
	    options.overwrite = true;
	    saveastiff(uint8(out), strcat(output_path, '/', tif_name), options);
	    imwrite(crop, strcat(validation_im_path, '/', strtok(tif_name, '.'), '.jpg'));

	    fprintf(['Image ', num2str(j), ' finished!\n']);
	end
end