--[[ 
Master's thesis
File description: Generates images and arranges them
Student: Ahmad Gheith
Supervisor: John Hallam
Date: 1 June 2018
]]--

require 'torch'
require 'nn'
require 'image'

function generate_noise_c(noise_c_dim, number_of_classes, batch_size, class)
	local noise = torch.randn(batch_size, noise_c_dim-number_of_classes, 1, 1)
	local class = class or torch.Tensor(batch_size, number_of_classes, 1, 1):uniform(0,1)
	if number_of_classes == 0 then; return noise; end;
	local noise_c = torch.cat(noise, class, 2)
	return noise_c, class, noise
end

function generate(netG, nrows, ncols, number_of_classes, class)
	local nimages = nrows*ncols
	local noise_c, classes = generate_noise_c(netG:get(1).nInputPlane, number_of_classes, nimages, class)
	if netG:get(1).weight:type() == 'torch.CudaTensor' then; noise_c = noise_c:cuda(); end
	local imgs = netG:forward(noise_c)
	return arrange(imgs, nrows, ncols), classes
end

function arrange(images, r, c, counter)
	if counter == nil then; counter = 1; end

	if r == 1 and c == 1 then; return images[counter]; end
	
	if r > 1 then
		return torch.cat(arrange(images, r-1, c, counter+c), arrange(images, 1, c, counter), 2)
	end

	if c > 1 then
		return torch.cat(arrange(images, r, c-1, counter+1), images[counter], 3)
	end

end

