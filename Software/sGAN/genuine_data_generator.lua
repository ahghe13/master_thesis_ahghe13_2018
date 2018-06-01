--[[ 
Master's thesis
File description: Generates genuine data for GAN/sGAN
Student: Ahmad Gheith
Supervisor: John Hallam
Date: 1 June 2018
]]--

require 'torch'
require 'image'

require 'libs/image_normalization'
require 'libs/tif_handling'
require 'libs/table_handling'


--######################### DEVIATION FUNCTION ##########################--

function image_generator(ideal, std_dev)
	local output = ideal:clone()
	for c=1,ideal:size(1) do
		for i=1,ideal:size(2) do
			for j=1,ideal:size(3) do
				output[c][i][j] = ideal[c][i][j] * torch.normal(1, std_dev)
			end
		end
	end
	return output
end


--########################### CLASS FUNCTIONS ###########################--

brightness = function(img)
	local b = torch.uniform(0,1)
	for c=1,img:size(1) do
		img[c][2][2] = img[c][2][2] * (b*3); img[c][2][3] = img[c][2][3] * (b*3)
		img[c][3][2] = img[c][3][2] * (b*3); img[c][3][3] = img[c][3][3] * (b*3)
	end
	return b
end


--############################### OPTIONS ###############################--

opt = {
	ideal_img_path = '',		-- Path to ideal image
	save_data_path = '',		-- Path to save the data set
	data_size = 2000,			-- Desired data size
	std_dev = 0.5				-- Standard deviation from the ideal image
	classes = {{'brightness', brightness}},	-- Classes to incorporate, none for GAN
}


--############################ GENERATE DATA #############################--

-- Prepare data_info
local data_info = {'file'}
for i=1,table.getn(opt.classes) do
	table.insert(data_info, opt.classes[i][1])
end
data_info = {data_info}

-- Load ideal image
local ideal = load_tif(opt.ideal_img_path, 'all', 'zero2one')

-- Generate data
for i=1,opt.data_size do
	local img_info = {i .. '.tif'}
	local img = image_generator(ideal, opt.std_dev)
	for i=1,table.getn(opt.classes) do
		class_value = opt.classes[i][2](img)
		table.insert(img_info, class_value)
	end
	table.insert(data_info, img_info)
	save_tif(opt.save_data_path .. '/' .. i .. '.tif', img)
end

Table2CSV(data_info, opt.save_data_path .. '/data_info.csv')
