% Master's thesis
% File description: Produces histogram based fingerprints of data images
% Student: Ahmad Gheith
% Supervisor: John Hallam
% Date: 1 June 2018

clear;
addpath('../tiff_handling');

tif_data_path = [];		% One or more comma separated paths to data

for i=1:size(tif_data_path,1)
	data_path = tif_data_path(i,:)

	l = dir(strcat(data_path, '*.tif'));

	[files, c] = size(l);

	output_path = strcat(data_path, 'histogram_prints');
	mkdir(output_path);

	fprintf(['\nProcessing images in ', output_path, '\n'])

	for j=1:files
	    tif_name = l(j).name;
	    
	    tif = loadtiff(strcat(data_path, tif_name));
	    out = HistPrint(tif, 64);
	    tif_name = strsplit(tif_name, '.'); tif_name = tif_name{1};
	    out = imresize(out, [64, 64]);
	    imwrite(out, strcat(output_path, '/', tif_name, '.jpg'));
	    fprintf(['Image ', num2str(j), ' finished!\n'])
	end
end