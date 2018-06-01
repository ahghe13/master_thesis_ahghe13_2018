% Master's thesis
% File description: Slices the data
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

clear;
addpath('../tiff_handling');

slice_dim = 64;

root_path = ''		% Root path to data

input_dirs = [];	% One or more paths to data
output_dirs = [];	% Similar number of paths to save the output

for i=1:size(input_dirs,1)
	data_path = strcat(root_path, input_dirs(i,:));

	l = dir(strcat(data_path, '*.tif'));

	[files, c] = size(l);

	output_path = strcat(root_path, output_dirs(i,:), '_', num2str(slice_dim), 'x', num2str(slice_dim), '/');
	mkdir(output_path);

	validation_im_path = strcat(output_path, '/slice_validation');
	mkdir(validation_im_path);

	fprintf(['\nProcessing images in ', data_path, '\n']);

	for j=1:files
	    tif_name = l(j).name;
	    
	    tif = loadtiff(strcat(data_path, tif_name));

	    [out, slice_valid] = image_slicer(tif, slice_dim);

	    clear options;
	    for k=1:size(out,4)
		    options.overwrite = true;
		    if k < 10
			    saveastiff(uint8(out(:,:,:,k)), strcat(output_path, '/', strtok(tif_name, '.'), 's00', num2str(k), '.tif'), options);
			elseif k < 100 
			    saveastiff(uint8(out(:,:,:,k)), strcat(output_path, '/', strtok(tif_name, '.'), 's0', num2str(k), '.tif'), options);
			else
			    saveastiff(uint8(out(:,:,:,k)), strcat(output_path, '/', strtok(tif_name, '.'), 's0', num2str(k), '.tif'), options);
			end
		end
	    imwrite(slice_valid, strcat(validation_im_path, '/', strtok(tif_name, '.'), '.jpg'));

	    fprintf(['Image ', num2str(j), ' finished!\n']);
	end
end