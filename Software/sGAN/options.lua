--[[ 
Master's thesis
File description: Options used by main.lua
Student: Ahmad Gheith
Supervisor: John Hallam
Date: 1 June 2018
]]--

--######################## GAN TYPE ########################--

net_size = 'full_size'			-- Choose between 'mini_size' or 'full_size'


--################# OPTIONS FOR MINI_cGAN ###################--

opt_mini_size = {
	net_name = 'mini',
	data_info = '',				-- Path to data_info file
	trainP = 80, testP = 10, validP = 10,

	epochs = 30,				-- Training Epochs
	batchSize = 12,				-- Batch Size
	imDim = 4,					-- Dimension of the image
	cs = {'all'},				-- Channel select
	nz = 100,					-- Number of noise elements passed to generator
	learningRate = 0.0005,  	-- Learning rate for Adam
	beta1 = 0.9,				-- Beta1
	beta2 = 0.999,				-- Beta2

	save_nets = 1,				-- Save every nth network; 0=disable
	save_nets_path = '',		-- Path to save the GAN / sGAN
	gpu = 0
}


--#################### OPTIONS FOR GAN/sGAN #####################--

opt_full_size = {
	net_name = 'full size',
	data_info = '',			-- Path to data info
	trainP = 80, testP = 10, validP = 10,

	epochs = 30,			-- Training Epochs
	batchSize = 12,			-- Batch Size
	imDim = 64,				-- Dimension of the image
	cs = {163},				-- Channel select. For all channels, write {'all'}
	nz = 200,				-- Number of noise elements passed to generator

	learningRate = 0.0002,	-- Learning rate for Adam
	beta1 = 0.5,			-- Beta1
	beta2 = 0.999,			-- Beta2

	save_nets = 1,			-- Save every nth network; 0=disable
	save_nets_path = '',	-- Path to save the GAN / sGAN
	gpu = 0
}
