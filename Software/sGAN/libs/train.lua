--[[ 
Master's thesis
File description: Training algorithm for GAN/sGAN
Student: Ahmad Gheith
Supervisor: John Hallam
Date: 1 June 2018
]]--

require 'torch'
require 'nn'
require 'optim'

include('data.lua')

function round(num, numDecimalPlaces)
  if numDecimalPlaces and numDecimalPlaces>0 then
    local mult = 10^numDecimalPlaces
    return math.floor(num * mult + 0.5) / mult
  end
  return math.floor(num + 0.5)
end

function Train(G, D, trainData, opt, e)
   local data = Data:create(trainData, opt.batchSize, opt.cs)

   local optimStateG = {learningRate = opt.learningRate, beta1 = opt.beta1, beta2 = opt.beta2}
   local optimStateD = {learningRate = opt.learningRate, beta1 = opt.beta1, beta2 = opt.beta2}

   local fake_label = 0
   local real_label = 1
   local inputD = torch.Tensor(opt.batchSize, table.getn(opt.cs), opt.imDim, opt.imDim)

   local inputG = torch.Tensor(opt.batchSize, opt.nz + data:getClasses(), 1, 1)
   local noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
   local class = torch.Tensor(opt.batchSize, data:getClasses())

   local labels = torch.Tensor(opt.batchSize, data:getClasses() + 1)
   local label = torch.Tensor(opt.batchSize, 1)

   local errD, errG

   local criterion = nn.BCECriterion()

   local epoch_tm = torch.Timer()
   local itr_tm = torch.Timer()
   local total_tm = torch.Timer()

   if opt.gpu > 0 then
      require 'cunn'
      cutorch.setDevice(opt.gpu)
      noise = noise:cuda(); class = class:cuda(); label = label:cuda(); labels = labels:cuda();
      inputD = inputD:cuda(); inputG = inputG:cuda();
      D:cuda(); G:cuda();
      criterion:cuda()
      print('GPU activated!')
   end

   local parametersD, gradParametersD = D:getParameters()
   local parametersG, gradParametersG = G:getParameters()

   local fDx = function(x)
      gradParametersD:zero()

      -- train with real
      local real, class_tmp = data:getBatch()
      class:copy(class_tmp)
      inputD:copy(real)
      label:fill(real_label)

      if data:getClasses() > 0 then; labels = torch.cat(label, class, 2); else; labels:copy(label); end;

      local output = D:forward(inputD)
      local errD_real = criterion:forward(output, labels)
      local df_do = criterion:backward(output, labels)
      D:backward(inputD, df_do)

      -- train with fake
      noise:normal()
      class:uniform(0,1)
      if data:getClasses() > 0 then; inputG = torch.cat(noise, class, 2); else; inputG:copy(noise); end;

      local fake = G:forward(inputG)
      inputD:copy(fake)
      label:fill(fake_label)
      if data:getClasses() > 0 then; labels = torch.cat(label, class, 2); else; labels = label; end;

      local output = D:forward(inputD)
      local errD_fake = criterion:forward(output, labels)
      local df_do = criterion:backward(output, labels)
      D:backward(inputD, df_do)

      errD = errD_real + errD_fake

      return errD, gradParametersD
   end

   -- create closure to evaluate f(X) and df/dX of generator
   local fGx = function(x)
      gradParametersG:zero()

      label:fill(real_label) -- fake labels are real for generator cost
      if data:getClasses() > 0 then; labels = torch.cat(label, class, 2); else; labels:copy(label); end;

      local output = D.output -- netD:forward(input) was already executed in fDx, so save computation
      errG = criterion:forward(output, labels)
      local df_do = criterion:backward(output, labels)
      local df_dg = D:updateGradInput(inputD, df_do)
      G:backward(inputG, df_dg)

      return errG, gradParametersG
   end

   local epoch = epoch
   if epoch == nil then; epoch = 1; end

   local totalBatches = data:getTotalBatches()
   total_tm:reset()
   errT = {}
   for i=1,e do
      epoch_tm:reset()
      print("Epoch " .. epoch)

      for j=1,totalBatches do
         itr_tm:reset()
         optim.adam(fDx, parametersD, optimStateD)
         optim.adam(fGx, parametersG, optimStateG)
         print("Epoch " .. epoch .. " (" .. j .. "/" .. totalBatches .. ")", 
            "Itr time: " .. round(itr_tm:time().real, 4) .. "s",
            "Total time: " .. round(total_tm:time().real, 4) .. "s",
            "errD: " .. round(errD, 4), "errG: " .. round(errG,4), 
            "Total Error: " .. round(errG + errD,4))
         table.insert(errT, {errG + errD})
      end

      data:shuffle()

      print("Epoch " .. epoch .. " successfully completed! Epoch time consumption: " 
         .. round(epoch_tm:time().real, 4) .. "s" .. "\n")

      if epoch % opt.save_nets == 0 then
         torch.save(opt.save_nets_path .. '/epoch' .. epoch .. '_netG.t7', netG:clearState())
         torch.save(opt.save_nets_path .. '/epoch' .. epoch .. '_netD.t7', netD:clearState())
      end


      epoch = epoch+1
   end
   Table2CSV(errT, opt.save_nets_path .. '/error.csv')
   print('Total time: ' .. total_tm:time().real);

end

