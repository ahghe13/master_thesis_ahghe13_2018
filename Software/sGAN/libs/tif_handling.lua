--[[ 
Master's thesis
File description: Help functions for easy tiff handling
Student: Ahmad Gheith
Supervisor: John Hallam
Date: 1 June 2018
]]--

require 'torch'
require 'image'

gm = require 'graphicsmagick'
include('image_normalization.lua')

function load_tif(path, channels, normalize)
	if channels == nil or channels == 'all' then
		c = io.popen('identify "' .. path .. '" | wc -l'); c = c:read()
		channels = {}; for i=1,c do;table.insert(channels, i-1); end
	end

	local sample = gm.Image(path .. '[0]'):toTensor('byte', 'I')
	local img = torch.Tensor(#channels, sample:size(1), sample:size(2))

	for i=1,#channels do
		img[i] = gm.Image(path .. '[' .. channels[i] .. ']'):toTensor('byte', 'I')
	end

	if normalize == 'minusone2one' then; img = norm_minusone2one(img);
	elseif normalize == 'zero2one' then; img = norm_zero2one(img);
	end
	
	return img, channels
end

function LoadImgs(paths, channels, normalize)
	-- Loads multiple images at once using paths in a table
	local img = load_tif(paths[1], channels, normalize)
	local imgs = torch.Tensor(#paths, #channels, img:size(2), img:size(3))
	imgs[1] = img:clone()
	for i=2,#paths do
		imgs[i] = load_tif(paths[i], channels, normalize)
	end
	return imgs
end

function save_tif(path, img)
	local im = norm_zero2one(img)
	local channels = im:size(1)
	local s = 'convert '
	local r = 'rm '
	for i=1,channels do
		image.save('temp_im'.. i .. '.jpg', im[i])
		s = s .. 'temp_im'.. i .. '.jpg '
		r = r .. 'temp_im'.. i .. '.jpg '
	end
	s = s .. '-colorspace Gray -adjoin "' .. path .. '"'
	os.execute(s)
	os.execute(r)
--	convert im.jpg im1.jpg -adjoin output.tif
end
