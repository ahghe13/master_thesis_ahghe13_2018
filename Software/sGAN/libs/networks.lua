--[[ 
Master's thesis
File description: The architectures of the networks
Student: Ahmad Gheith
Supervisor: John Hallam
Date: 1 June 2018
]]--

require 'torch'
require 'nn'

include('weights_init.lua')


function mini_networks(opt)
	--######################### GENERATOR ########################--

	local netG = nn.Sequential()
	-- Input consists of noise vector, z, and class parameter(s)
	netG:add(nn.SpatialFullConvolution(opt.nz + table.getn(opt.classes), 4, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(4)):add(nn.ReLU(true))
	-- State size: 4 x 2 x 2
	netG:add(nn.SpatialFullConvolution(4, table.getn(opt.cs), 4, 4, 2, 2, 1, 1))
	netG:add(nn.Tanh())
	-- state size: C x 4 x 4
	netG:apply(weights_init)

	--####################### DESCRIMINATOR #######################--

	local netD = nn.Sequential()
	-- input is C x 4 x 4
	netD:add(nn.SpatialConvolution(table.getn(opt.cs), 4, 4, 4, 2, 2, 1, 1))
	netD:add(nn.LeakyReLU(0.2, true))
	-- state size: 4 x 2 x 2
	netD:add(nn.SpatialConvolution(4, table.getn(opt.classes) + 1, 4, 4, 2, 2, 1, 1))
	netD:add(nn.Sigmoid())
	-- state size: (classes + 1) x 1 x 1
	netD:add(nn.View(table.getn(opt.classes) + 1):setNumInputDims(3))
	-- state size: (classes + 1)
	netD:apply(weights_init)

	return netG, netD

end

function networks(opt)
	--######################### GENERATOR ########################--

	local netG = nn.Sequential()
	-- Input consists of noise vector, z, and class parameter(s)
	netG:add(nn.SpatialFullConvolution(opt.nz + table.getn(opt.classes), 512, 4, 4))
	netG:add(nn.SpatialBatchNormalization(512)):add(nn.ReLU(true))
	-- state size: 512 x 4 x 4
	netG:add(nn.SpatialFullConvolution(512, 256, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(256)):add(nn.ReLU(true))
	-- state size: 256 x 8 x 8
	netG:add(nn.SpatialFullConvolution(256, 128, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(128)):add(nn.ReLU(true))
	-- state size: 128 x 16 x 16
	netG:add(nn.SpatialFullConvolution(128, 64, 4, 4, 2, 2, 1, 1))
	netG:add(nn.SpatialBatchNormalization(64)):add(nn.ReLU(true))
	-- state size: 64 x 32 x 32
	netG:add(nn.SpatialFullConvolution(64, table.getn(opt.cs), 4, 4, 2, 2, 1, 1))
	netG:add(nn.Tanh())
	-- state size: C x 64 x 64

	netG:apply(weights_init)


	--####################### DESCRIMINATOR #######################--

	local netD = nn.Sequential()
	-- input is C x 64 x 64
	netD:add(nn.SpatialConvolution(table.getn(opt.cs), 64, 4, 4, 2, 2, 1, 1))
	netD:add(nn.LeakyReLU(0.2, true))
	-- state size: 64 x 32 x 32
	netD:add(nn.SpatialConvolution(64, 128, 4, 4, 2, 2, 1, 1))
	netD:add(nn.SpatialBatchNormalization(128)):add(nn.LeakyReLU(0.2, true))
	-- state size: 128 x 16 x 16
	netD:add(nn.SpatialConvolution(128, 256, 4, 4, 2, 2, 1, 1))
	netD:add(nn.SpatialBatchNormalization(256)):add(nn.LeakyReLU(0.2, true))
	-- state size: 256 x 8 x 8
	netD:add(nn.SpatialConvolution(256, 512, 4, 4, 2, 2, 1, 1))
	netD:add(nn.SpatialBatchNormalization(512)):add(nn.LeakyReLU(0.2, true))
	-- state size: 512 x 4 x 4
	netD:add(nn.SpatialConvolution(512, table.getn(opt.classes) + 1, 4, 4))
	netD:add(nn.Sigmoid())
	-- state size: (classes + 1) x 1 x 1
	netD:add(nn.View(table.getn(opt.classes) + 1):setNumInputDims(3))
	-- state size: (classes + 1)

	netD:apply(weights_init)

	return netG, netD

end
