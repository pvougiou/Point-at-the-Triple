----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree.
----
----  Parts of this code are adapted from: https://github.com/wojzaremba/lstm/blob/master/main.lua
----

local params = {
    -- The path to the folder with all the required dataset-related files.
    dataset_path = '../Full/without_surf_form_tuples/',
    -- Periodically this file will be saving checkpoints of the trained model.
    checkpoint_dump_path = './checkpoints/full/without_surf_form_tuples/epoch_%.2f.error_%.4f.without_surf_form_tuples.model.t7',
    checkpoint = '',
    batch_size = 80,
    layers = 1,
    decay = 0.8,
    rnn_size = 500,
    dropout = 0.0,
    init_weight = 1e-1,
    learning_rate = 5e-5,
    max_epoch = 100,
    max_norm = 3,
    max_grad_norm = 10,
    gpuidx = 1,
    testing = false
}

package.path = './utils/?.lua;' .. package.path
local ok1, cunn = pcall(require, 'cunn')
local ok2, cutorch = pcall(require, 'cutorch')
if not (ok1 and ok2) then
    print('Warning: Either cunn or cutorch was not found. Falling gracefully to CPU...')
    params.gpuidx = 0
    pcall(require, 'nn')
else
    cutorch.setDevice(params.gpuidx)
    print(string.format("GPU: %d", params.gpuidx) .. string.format(' out of total %d is being currently used.', cutorch.getDeviceCount()))
end

require('string')
require('nngraph')
-- nngraph.setDebug(true)

require('optim')
require('utilities')
dataset = require('dataset')
require('LookupTableMaskZero')
require('MaskedClassNLLCriterionInheritingNLLCriterion')


local function gru(x, prev_h)
    local i2h			= nn.Linear(params.rnn_size, 2 * params.rnn_size)(x)
    local h2h			= nn.Linear(params.rnn_size, 2 * params.rnn_size)(prev_h)
    local gates			= nn.CAddTable()({i2h, h2h})
    
    local reshapedGates		= nn.Reshape(2, params.rnn_size)(gates)
    local splitGates		= nn.SplitTable(2)(reshapedGates)
    
    local reset_gate		= nn.Sigmoid()(nn.SelectTable(1)(splitGates))
    local update_gate		= nn.Sigmoid()(nn.SelectTable(2)(splitGates))
    local candidate_h		= nn.Tanh()(nn.CAddTable()({
						    nn.Linear(params.rnn_size, params.rnn_size)(x),
						    nn.Linear(params.rnn_size, params.rnn_size)(nn.CMulTable()({reset_gate, prev_h}))}))
    
    local next_h		= nn.CAddTable()({nn.CMulTable()(
						      {nn.AddConstant(1, false)(
							   nn.MulConstant(-1, false)(
							       update_gate)), prev_h}),
						  nn.CMulTable()({update_gate, candidate_h})})
    return next_h
end


local function create_encoder()
    local x			= nn.Identity()() 
    local i			= {}
    i[0]			= nn.Reshape(params.num_aligned_triples * params.batch_size, 3 * params.rnn_size, false)(
	nn.LookupTableMaskZero(params.source_vocab_size, params.rnn_size)(x))
    local n2i			= nn.BatchNormalization(params.rnn_size, 1e-5, 0.1, true)
    i[1]			= nn.ReLU()(n2i(nn.Linear(3 * params.rnn_size, params.rnn_size, false)(
						    nn.Dropout(params.dropout)(i[0]))))
    for layeridx = 2, params.layers do
	i[layeridx] = nn.ReLU()(nn.BatchNormalization(params.rnn_size, 1e-5, 0.1, true)(
				    nn.Linear(params.rnn_size, params.rnn_size, false)(
					nn.Dropout(params.dropout)(i[layeridx - 1]))))
    end
    local tripleEmbeddings	= nn.SplitTable(1)(
	{nn.Reshape(params.num_aligned_triples, params.batch_size, params.rnn_size, false)(i[params.layers])})

    local module                = nn.gModule({x}, {nn.Identity()(tripleEmbeddings)})  
    return module

end


local function create_decoder()
    local x			= nn.Identity()()
    local prev_state		= nn.Identity()()
    local source_states         = nn.Identity()()

    local attn_mask             = nn.Identity()()
    local surf_form_mask        = nn.Identity()()  
    
    local splitSourceStates     = {source_states:split(params.num_aligned_triples)}
    local i			= {}

    i[0]			= nn.LookupTableMaskZero(params.target_vocab_size, params.rnn_size)(x)
    local next_state		= {}
    if params.layers == 1 then
	local prev_h		= prev_state
	local next_h		= gru(i[0], prev_h)
	table.insert(next_state, next_h)
	i[1] = next_h
    else
	local splitPrevState	= {prev_state:split(params.layers)}
	for layeridx = 1, params.layers do
	    local prev_h	= splitPrevState[layeridx]
	    local next_h        = gru(i[layeridx - 1], prev_h)
	    table.insert(next_state, next_h)
	    i[layeridx] = next_h
	end
    end
    local scores                = {}
    
    -- Inspired from: https://github.com/OpenNMT/OpenNMT/blob/28510242fc6bfcca4b0fc4a507209edbccb78b2b/onmt/modules/GlobalAttention.lua
    local weighted_h            = nn.Linear(params.rnn_size, params.rnn_size)(i[params.layers])
    for tripleidx = 1, params.num_aligned_triples do
    	local source_triple     = splitSourceStates[tripleidx]
    	table.insert(scores, nn.Reshape(1)(nn.DotProduct()({source_triple, weighted_h})))
    end
    
    local attn_distr            = nn.SoftMax()(nn.CMulTable()({nn.JoinTable(2)(scores), attn_mask}))
    local context               = nn.MixtureTable()({attn_distr, source_states})

    local h1h                   = nn.Linear(2 * params.rnn_size, params.rnn_size)(
	nn.Dropout(params.dropout)(
	    nn.JoinTable(2)({i[params.layers], context})))

    
    local h1y                   = nn.Linear(params.rnn_size, params.target_vocab_size)
    local dropped               = nn.Dropout(params.dropout)

    -- Similar to (See et al. 2017) [https://arxiv.org/abs/1704.04368]
    local pred_vocab            = nn.SoftMax()(h1y(dropped(h1h)))
    local pred_gen              = nn.Reshape(true)(nn.Sigmoid()(
    						       nn.CAddTable()(
    							   {nn.BatchNormalization(1, 1e-5, 0.1, true)(nn.Linear(params.rnn_size, 1)(context)),
    							    nn.BatchNormalization(1, 1e-5, 0.1, true)(nn.Linear(params.rnn_size, 1)(i[params.layers]))})))
    
    splitPredForm              = {}
    for tripleidx = 1, params.num_aligned_triples do
	print(tripleidx)
	local surf_form_scores = nn.Replicate(params.num_surf_forms, 2)(nn.Reshape(true)(scores[tripleidx]))
	table.insert(splitPredForm, surf_form_scores)
    end


    local pred_form          = nn.CMulTable()({
    	    nn.Replicate(params.num_aligned_triples * params.num_surf_forms , 2)(
    		nn.AddConstant(1, false)(nn.MulConstant(-1, false)(pred_gen))),
	    nn.SoftMax()(nn.CMulTable()({nn.JoinTable(2)(splitPredForm), surf_form_mask}))})


    local pred               = nn.Log()(nn.CAddTable()({
					nn.CMulTable()({nn.Replicate(params.target_vocab_size, 2)((pred_gen)), pred_vocab}),
					nn.Padding(1, - (params.target_vocab_size - (params.num_surf_forms * params.num_aligned_triples)), 1, 0)(pred_form)}))
    
    local module             = nn.gModule({x, prev_state, source_states, attn_mask, surf_form_mask},
	{pred, nn.Identity()(next_state)})

    return module

end


local function setup(model)
    if string.len(params.checkpoint) > 0 then
        print("Loading a Triples2GRU network...")
        encoder, decoder = model.encoder, model.decoder
        x, dx = combine_all_parameters(encoder.network, decoder.network)
        print (x:size())
        print (dx:size())
    else
        print("Creating a Triples2GRU network...")
        local encoderNetwork = transfer_to_gpu(create_encoder(), params.gpuidx)
        local decoderNetwork = transfer_to_gpu(create_decoder(), params.gpuidx)
	
        x, dx = combine_all_parameters(encoderNetwork, decoderNetwork)
        x:uniform(-params.init_weight, params.init_weight)
        print (x:size())
        print (dx:size())
	
        encoder = {}
        decoder = {}
	encoder.network = encoderNetwork
        decoder.network = decoderNetwork
	decoder.rnns = cloneManyTimes(decoderNetwork, params.timesteps)
    end
    
    decoder.s = {}
    decoder.pred = {}
    decoder.ds = {}

    encoder.s = {}
    encoder.ds = {}
    
    for j = 1, params.num_aligned_triples do
	encoder.s[j] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	encoder.ds[j] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
    end


    for j = 0, params.timesteps do
	decoder.s[j] = {}
	if params.layers == 1 then
	    decoder.s[j] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	else
	    for d = 1, params.layers do
		decoder.s[j][d] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	    end
	end
    end

    for j = 0, params.timesteps do
        decoder.pred[j] = transfer_to_gpu(torch.zeros(params.batch_size, params.target_vocab_size), params.gpuidx)
    end
    
    if params.layers == 1 then
	decoder.ds = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)	
    else
	for d = 1, params.layers do
	    decoder.ds[d] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	end
    end
    
    local criterion = transfer_to_gpu(nn.MaskedClassNLLCriterion(), params.gpuidx)
    decoder.criterion = criterion
    decoder.criterions = cloneManyTimes(criterion, params.timesteps)
    
    decoder.err = transfer_to_gpu(torch.zeros(params.timesteps), params.gpuidx)

    

    decoder.attn_mask = transfer_to_gpu(torch.DoubleTensor(), params.gpuidx)
    decoder.surf_form_mask = transfer_to_gpu(torch.DoubleTensor(), params.gpuidx)

    collectgarbage()
    collectgarbage()
end


local function reset_state(state)
    state.batchidx = 1
    print('State: ' .. string.format("%s", state.name) .. ' has been reset.')
end



local function forward(state, x_new)
    if x ~= x_new then x:copy(x_new) end
    
    if state.batchidx > state.triples:size(1) then reset_state(state) end

    local batchTriples = state.triples[{state.batchidx, {}, {}, {}}]:reshape(params.batch_size * params.num_aligned_triples, 3)
    local attn_mask = state.triples[{state.batchidx, {}, {}, 1}]:transpose(1, 2):ne(0)

    local sub_mask = state.triples[{state.batchidx, {}, {}, 1}]:transpose(1, 2):le(params.num_incl_predicates + params.num_freq_entities + 1)
    local obj_mask = state.triples[{state.batchidx, {}, {}, 3}]:transpose(1, 2):le(params.num_incl_predicates + params.num_freq_entities + 1)

    
    -- Remove comment from the line below in case params.num_surf_forms == 1.
    local triple_to_surf_form_mask = torch.cmul(attn_mask, torch.cmul(sub_mask, obj_mask))
    -- if params.num_surf_forms == 1 then
    -- local surf_form_mask = torch.reshape(attn_mask:transpose(1,2),
    -- 					 params.num_surf_forms * params.num_aligned_triples, params.batch_size):transpose(1,2)


    -- if params.num_surf_forms == 2 then
    -- local surf_form_mask = torch.reshape(torch.cat(
    -- 					     attn_mask:transpose(1,2), triple_to_surf_form_mask:transpose(1,2)),
    -- 					 params.num_surf_forms * params.num_aligned_triples, params.batch_size):transpose(1,2)
    -- if params.num_surf_forms == 3 then
    local surf_form_mask = torch.reshape(torch.cat(
    					     torch.cat(attn_mask:transpose(1,2), triple_to_surf_form_mask:transpose(1,2)),
    					     triple_to_surf_form_mask:transpose(1,2)),
    					 params.num_surf_forms * params.num_aligned_triples, params.batch_size):transpose(1,2)
    



    encoder.s = encoder.network:forward(batchTriples)
    if params.gpuidx > 0 then cutorch.synchronize() end
    
    -- We initialise the attentive decoder.
    if params.layers == 1 then
        decoder.s[0]:zero()
    else
        for d = 1, #decoder.s[0] do
	    decoder.s[0][d]:zero()
        end
    end  


   
    decoder.attn_mask:resize(attn_mask:size()):copy(attn_mask) -- can reuse memory
    decoder.surf_form_mask:resize(surf_form_mask:size()):copy(surf_form_mask) -- can reuse memory
    
    for i = 1, params.timesteps do
        decoder.pred[i], decoder.s[i] = unpack(decoder.rnns[i]:forward({state.summaries[{state.batchidx, {}, i}],
									decoder.s[i - 1],
									encoder.s,
									decoder.attn_mask,
									decoder.surf_form_mask}))


	


	
	if params.gpuidx > 0 then cutorch.synchronize() end
	decoder.err[i] = decoder.criterions[i]:forward(decoder.pred[i], state.summaries[{state.batchidx, {}, i + 1}])
        if params.gpuidx > 0 then cutorch.synchronize() end
    end
    
    -- print_gpu_usage

    local validBatches = state.summaries[{state.batchidx, {}, {2, params.timesteps + 1}}]:ne(0):eq(1):sum(1)
    local mask = validBatches:ne(0)
    return torch.cdiv(decoder.err[mask]:contiguous():typeAs(decoder.err), validBatches[mask]:contiguous():typeAs(decoder.err)):mean()
	
end


local function backward(state)

    dx:zero()
    -- Reset gradients
    -- Gradients are always accumulated to accomodate batch methods.
    for d = 1, #encoder.ds do encoder.ds[d]:zero() end
    if params.layers == 1 then decoder.ds:zero()
    else
	for d = 1, #decoder.ds do decoder.ds[d]:zero() end
    end
    
	
    for i = params.timesteps, 1, -1 do
	local tempPrediction = decoder.criterions[i]:backward(decoder.pred[i],
							      state.summaries[{state.batchidx, {}, i + 1}])

	local _, tempDecoder, tempEncoder, _, _ = unpack(decoder.rnns[i]:backward({state.summaries[{state.batchidx, {}, i}],
									     -- state.summaries[{state.batchidx, {}, i + 1}],
									     decoder.s[i - 1],
									     encoder.s,
									     attn_mask_double},
							  {tempPrediction, decoder.ds}))
        
	
	if params.layers == 1 then decoder.ds:copy(tempDecoder)
	else copyTable(tempDecoder, decoder.ds) end
	for d = 1, #encoder.ds do encoder.ds[d]:add(tempEncoder[d]) end
	if params.gpuidx > 0 then cutorch.synchronize() end
    end

  
	
    for d = 1, #encoder.ds do
	assert(encoder.ds[d]:norm() > 1e-5)
    end
    local batchTriples = state.triples[{state.batchidx, {}, {}, {}}]:reshape(params.batch_size * params.num_aligned_triples, 3)
    encoder.network:backward(batchTriples, encoder.ds)
    if params.gpuidx > 0 then cutorch.synchronize() end
    state.batchidx = state.batchidx + 1

    -- Comment-in should you want to set a maximum threshold for the norm of the gradients.
    -- In case it is exceeded gradients are scaled down according to the shrink_factor.
    -- norm_dx = dx:norm()
    -- if norm_dx > params.max_grad_norm then
    -- 	local shrink_factor = params.max_grad_norm / norm_dw
    -- 	dx:mul(shrink_factor)
    -- end
end


local function feval(x_new)
    if params.testing then
	forward(testing, x_new)
	backward(testing)
    else
	forward(training, x_new)
	backward(training)
    end
    return 0, dx
end

-- Evaluate by computing perplexity on either the validation set.
local function evaluate(state)
    reset_state(state)
    	
    encoder.network:evaluate()
    decoder.network:evaluate()
    for j = 1, #decoder.rnns do decoder.rnns[j]:evaluate() end
    
    local perplexity = 0
    
    while state.batchidx <= state.triples:size(1) do
	perplexity = perplexity + forward(state, x)
	print (string.format("%d", state.batchidx).. '\t/ '.. string.format("%d", state.triples:size(1)))
	state.batchidx = state.batchidx + 1
	collectgarbage()
	collectgarbage()
    end
    perplexity = torch.exp(perplexity / state.triples:size(1))
    print(string.format('%s Set Perplexity: ', state.name) .. string.format("%.4f", perplexity))

    encoder.network:training()
    decoder.network:training()
    for j = 1, #decoder.rnns do decoder.rnns[j]:training() end
    return perplexity
end

local function main()
    
    sgd_params = {
	learning_rate = params.learning_rate,
	momentum = 0
    }

    rmsprop_params = {
	learning_rate = params.learning_rate,
	-- weightDecay = 0.1,
	alpha = 0.9
    }

    adagrad_params = {
	-- weightDecay = 0.1,
	-- learningRateDecay = 0.1
	learning_rate = params.learning_rate
    }

    adam_params = {
	weightDecay = 0.05,
	-- learningRateDecay = 0.1
	learning_rate = params.learning_rate,
	-- epsilon = 1e-2
    }
    
    -- Initialising the dataset.
    dataset.init_dataset(params.dataset_path)
    
    local triples_dictionary = dataset.triples_dictionary()
    local summaries_dictionary = dataset.summaries_dictionary()
    assert(length(triples_dictionary['item2id']) == length(triples_dictionary['id2item']))
    assert(length(summaries_dictionary['word2id']) == length(summaries_dictionary['id2word']))
    params.num_aligned_triples = triples_dictionary['max_num_triples']
    params.num_surf_forms = triples_dictionary['max_num_surf_forms']
    params.num_freq_entities = triples_dictionary['num_freq_entities']
    params.num_incl_predicates = triples_dictionary['num_incl_predicates']
    params.source_vocab_size = length(triples_dictionary['item2id']) - 1
    params.target_vocab_size = length(summaries_dictionary['word2id']) - 1
	
    training = {
	triples = transfer_to_gpu(dataset.train_triples(params.num_aligned_triples, params.batch_size), params.gpuidx),
	summaries = transfer_to_gpu(dataset.train_summaries(params.batch_size), params.gpuidx),
	name = 'Training'
    }
    validation = {
	triples = transfer_to_gpu(dataset.validate_triples(params.num_aligned_triples, params.batch_size), params.gpuidx),
	summaries = transfer_to_gpu(dataset.validate_summaries(params.batch_size), params.gpuidx),
	name = 'Validation'
    }
    testing = {
	triples = transfer_to_gpu(dataset.test_triples(params.num_aligned_triples, params.batch_size), params.gpuidx),
	summaries = transfer_to_gpu(dataset.test_summaries(params.batch_size), params.gpuidx),
	name = 'Testing'
    }
	
    print(training.summaries:size())
    print(training.triples:size())
    print(validation.summaries:size())
    print(validation.triples:size())
    print(testing.summaries:size())
    print(testing.triples:size())

    assert(training.summaries:size(3) == validation.summaries:size(3))
    assert(training.summaries:size(3) == testing.summaries:size(3))
    -- The number of timesteps that we unroll for.
    params.timesteps = training.summaries:size(3) - 1
	

    reset_state(training)
    reset_state(validation)
    reset_state(testing)

    local details = {
	params = params,
	epoch = 0, 
	err = 0
    }
 
    if string.len(params.checkpoint) > 0 then
	local checkpoint = torch.load(params.checkpoint)
	for k, v in pairs(checkpoint.details.params) do 
	    if k == 'batch_size' or k == 'layers' or k == 'rnn_size' then
		params[k] = v
	    end
	end
	-- Storing the learning rate with which the loaded model from the checkpoint had been trained.
	params.checkpoint_learning_rate = checkpoint.details.params[learning_rate]
	print('Network Parameters')
	print(params)
	setup(checkpoint.model)
	details.epoch = checkpoint.details.epoch
    else
	print('Network Parameters')
	print(params)
	setup({})
    end

    print_gpu_usage(params.gpuidx)
	
    
    local epoch_size = training.triples:size(1)
    if params.testing then epoch_size = testing.triples:size(1) end
    
    -- local words_per_step = params.batch_size * params.timesteps
    local step = details.epoch * epoch_size
    
    -- Comment-in in case you want to measure perplexity on the test set. 
    if params.testing then details.err = evaluate(testing) end
	
    validationErrors = {}
    timer = torch.Timer() -- the Timer starts to count now
    while details.epoch < params.max_epoch do
	optim.adam(feval, x, adam_params)
	step = step + 1
	details.epoch = step / epoch_size
	-- Stops the timer. The accumulated time counted until now is stored.
	timer:stop()
	print (string.format("%d", step).. '\t/ '.. string.format("%d", epoch_size).. ': '.. string.format('Training loss is %.2f.', decoder.err:sum()))
	if step % torch.round(epoch_size / 2) == 0 then
	    if params.testing then
		
		details.err = evaluate(testing)
		table.insert(validationErrors, details.err)
		print(string.format("We have been training for %.2f seconds.", timer:time().real))
		print('Below are the Testing Errors for every half training epoch until now: ')		
		print(validationErrors)
	    else
		details.err = evaluate(validation)
		table.insert(validationErrors, details.err)
		print(string.format("We have been training for %.2f seconds.", timer:time().real))
		print('Below are the Validation Errors for every half training epoch until now: ')		
		print(validationErrors)
		if details.epoch >= 6  then 
		    -- Need to come up with a solution for the clearing the state
		    -- of the decoder, but it's not so trivial since the criterion
		    -- is part of the decoder gModule. Let's say Future Work for now.
		    encoder.network:clearState()
		    for j = 1, #decoder.rnns do decoder.rnns[j]:clearState() end
		    local tempEncoder = {}
		    local tempDecoder = {}
		    tempEncoder.network = encoder.network
		    tempDecoder.network = decoder.network
		    tempDecoder.rnns = decoder.rnns
		    -- Saving a checkpoint file of the trained model in the ./checkpoints directory.
		    generateCheckpoint(params.checkpoint_dump_path, {encoder = tempEncoder, decoder = tempDecoder}, details)
		end
	    end
	end
	timer:resume()
	
	collectgarbage()
	collectgarbage()
    end
    print('Training has been completed.')
end

main()
