--[[ 
Master's thesis
File description: Constructs and trains a GAN/sGAN
Student: Ahmad Gheith
Supervisor: John Hallam
Date: 1 June 2018
]]--

require 'torch'
require 'nn'
require 'image'

include('options.lua')
include('libs/networks.lua')
include('libs/train.lua')
include('libs/table_handling.lua')

--##################### OPTIONS SETUP ######################--

if net_size == 'mini_size' then; opt = opt_mini_size; end
if net_size == 'full_size' then; opt = opt_full_size; end

opt.save_nets_path = arg[1] or opt.save_nets_path
opt.cs = arg[2] or opt.cs

if opt.display == 0 then; opt.display = false; else; opt.display = true; end

torch.manualSeed(79)

--#################### DATA PREPARATION #####################--
-- Convert csv to table; save classes in opt.classes
dataTable = CSV2Table(opt.data_info); opt.classes = cloneTable(dataTable[1])

-- Save classes to train, test, and valid files
Table2CSV({opt.classes}, opt.save_nets_path .. '/train.csv')
Table2CSV({opt.classes}, opt.save_nets_path .. '/test.csv')
Table2CSV({opt.classes}, opt.save_nets_path .. '/valid.csv')

-- Remove first row to only include data; Remove first col of classes, which is file path
table.remove(dataTable, 1); table.remove(opt.classes, 1)

CatLine(opt.data_info:match(".*/"), dataTable, 'all', 1)

dataTable = Shuffle(dataTable)
dataP = SplitTable(dataTable, {opt.trainP, opt.testP, opt.validP})
train = dataP[1]; test = dataP[2]; valid = dataP[3]

Table2CSV(train, opt.save_nets_path .. '/train.csv', 'a')
Table2CSV(test, opt.save_nets_path .. '/test.csv', 'a')
Table2CSV(valid, opt.save_nets_path .. '/valid.csv', 'a')

if opt.cs[1] == 'all' then; local im; im, opt.cs = load_tif(train[1][1]); end

print(opt)
torch.save(opt.save_nets_path .. '/opt.t7', opt)

--########### DECLARE GENERATOR & DESCRIMINATOR ##############--

if net_size == 'mini_size' then; netG, netD = mini_networks(opt)
elseif net_size == 'full_size' then; netG, netD = networks(opt); end


--######################### TRAINING #########################--

Train(netG, netD, train, opt, opt.epochs)
