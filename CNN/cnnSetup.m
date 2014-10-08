function [cnn, W, b, Z] = cnnSetup()
%%CNNSETUP 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 19 Sep 2014
    
%   Time-stamp: <2014-10-08 11:22:24 by xenuts>
   
    %% Options for some certain layers
    NormOpts_localcn = struct(...
        'Kernel', fspecial('gaussian', [5 5], 2), ...
        'PadType', 'reflect' ... % 'reflect', 'zero'
        );
%     NormOpts_localcn = {struct2cell(NormOpts_localcn)};
    
    %% Define all layers
% % %     cnn.Layers = { ...
% % %         struct('Type', 'i', ... % input layer
% % %                'NormType', 'local_cn',...
% % %                'NormOpts', NormOpts_localcn), ... 
% % %         ... %%======================== MAJOR LAYER 1 =========================
% % %         struct('Type', 'c',...
% % %                'ConnType', 'full', ... % convlution layer
% % %                'KernelSize', [5 5], ...
% % %                'NumOut', 4, ...
% % %                'NonLinearity', 'relu'), ... 
% % %         struct('Type', 'p',...
% % %                'PoolingType', 'Mean', ... % pooling layer
% % %                'PoolingSize', [2 2], 'Stride', [2 2], ... 
% % %                'MAX_ABS', 1), ...
% % %         struct('Type', 'o', ... % output layer
% % %                'Method', 'softmax', ... %'softmax', 'sigmoid', 'linear'
% % %                'SizeOut', 10)... % When 'softmax', SizeOut = NumClass; when 'sigmoid', SizeOut = 1
% % %                  };

    %% For Tracking(regression)
% % %     cnn.Layers = { ...
% % %         struct('Type', 'i', ... % input layer
% % %                'NormType', 'local_cn',...
% % %                'NormOpts', NormOpts_localcn), ... 
% % %         ... %%======================== MAJOR LAYER 1 =========================
% % %         struct('Type', 'c',...
% % %                'ConnType', 'full', ... % convlution layer
% % %                'KernelSize', [5 5], ...
% % %                'NumOut', 4, ...
% % %                'NonLinearity', 'relu'), ... 
% % %         struct('Type', 'p',...
% % %                'PoolingType', 'Mean', ... % pooling layer
% % %                'PoolingSize', [2 2], 'Stride', [2 2], ... 
% % %                'MAX_ABS', 1), ...
% % %         struct('Type', 'o', ... % output layer
% % %                'Method', 'sigmoid', ... %'softmax', 'sigmoid', 'linear'
% % %                'SizeOut', 6)... % When 'softmax', SizeOut = NumClass; when 'sigmoid', SizeOut = 1
% % %                  };
             
        %% One Conv+Pool Layer (Works well)
                 cnn.Layers = { ...
        struct('Type', 'i', ... % input layer
               'NormType', 'none',... %'none', 'local_cn'
               'NormOpts', NormOpts_localcn...
               ), ...
        struct('Type', 'c',...
               'ConnType', 'full', ... % convlution layer
               'KernelSize', [5 5], ...
               'NumOut', 1, ...
               'NonLinearity', 'relu'), ... %sigmoid, tanh, relu(sometime unconvergence,why??)
        struct('Type', 'p',...
               'PoolingType', 'Mean', ... % pooling layer,'Max', 'Mean'
               'PoolingSize', [2 2], 'Stride', [2 2], ...
               'MAX_ABS', 1), ...
        struct('Type', 'o', ... % output layer
               'Method', 'softmax', ... %'softmax', 'sigmoid', 'linear'
               'SizeOut', 10)... % When 'softmax', SizeOut = NumClass; when 'sigmoid', SizeOut = 1
                 };
             
% % %                      struct('Type', 'i', ... % input layer
% % %                'NormType', 'local_cn',...
% % %                'NormOpts', NormOpts_localcn), ...
             
% % %         struct('Type', 'f',...
% % %                'NonLinearity', 'sigmoid', ... % full-connected layer
% % %                'SizeOut', 5 ... %'same'
% % %                ), ...
% %               
% % %         struct('Type', 'c',...
% % %                'ConnType', 'full', ... % convlution layer
% % %                'KernelSize', [5 5], ...
% % %                'NumOut', 4, ...
% % %                'NonLinearity', 'relu'), ... 
% % %         struct('Type', 'p',...
% % %                'PoolingType', 'Mean', ... % pooling layer,'Max', 'Mean'
% % %                'PoolingSize', [2 2], 'Stride', [2 2], ...
% % %                'MAX_ABS', 1), ...
%         ... %%======================== MAJOR LAYER 1 =========================
%         struct('Type', 'c',...
%                'ConnType', 'full', ... % convlution layer
%                'KernelSize', [5 5], ...
%                'NumOut', 8, ...
%                'NonLinearity', 'relu'), ... 
%         struct('Type', 'p',...
%                'PoolingType', 'Mean', ... % pooling layer
%                'PoolingSize', [2 2], 'Stride', [2 2], ...
%                'MAX_ABS', 1), ...  

%         struct('Type', 'f',...
%            'NonLinearity', 'sigmoid', ... % full-connected layer
%            'SizeOut', 'same' ...
%            ), ... 
    %% CNN general options
    cnn.Opts = struct(...
        'BatchSize', 100, ... %Qs: when dataset larger, larger or smaller BatchSize get good performance(quick convergent), WHY? SGD???
        'NumEpoch', 300, ...
        'RandomShuffle', 1, ...
        'Alpha', 9e-1, ...
        'MinAlpha', 1e-2, ... 
        'Momentum', 0, ...
        'Lambda', 0, ... %3e-3 factor of regularization cost, turn off if set 0
        'Dropout', 0.5, ... % Dropout fraction factor for full layers [0, 1)
        'CHECK_GRAD', 0 ... % When check gradient, better to turn off Dropout
        );
    
    %% Runtime Data (pre-allocate memory)
    % For a 'c' layer L, W{L} are the filters for convolution
    % For an 'f' or 'o' layer L, W{L} are the weights. Keep [] for other non-conv layers
    % For 'i' and 'p' layers, W{L}=[].
    W = cell(1, numel(cnn.Layers));
    b = cell(1, numel(cnn.Layers));
    
    % Activations or known as Feature Maps for each layer.
    % Z{1} = input
    Z = cell(1, numel(cnn.Layers));    
end

%% Testing on MNIST-N20
% i -> o                         [X] quick
% i -> c -> p -> o               [X] quick
% i -> f -> o                    [X] quick
% i -> c -> f -> o               [X] slow and unstable in early epoches
% i -> c -> p -> f -> o          [X] normal and unstable in early epoches
% i -> c -> p -> f -> f -> o     [X] very slow and unstable, not convergent at all when hidden layer's size is large
%                                    Dropout will help to slow convergent       
% Max => slow convergence, Mean => quick convergence
%% Tentative Conclusion:
% 1. Without dropout, too large batchsize usually cause very slow and unstable convergence
% 2. Dropout help training even when batchsize is too large. 
% 3. In one word, batchsize is crucial especially when no using dropout.
% 4. Too large size of hidden full layers cause slow and unstable convergence in early epoches
% 5. When training on large dataset, learning rate need to be scheduley decreasing to pursuit convergence

%% Tips
% 1. see http://cs.stanford.edu/people/karpathy/convnetjs/docs.html
% 1.1 ReLU is the best activation to use if you know nothing about these networks. However, you have to be careful with keeping learning rates small because ReLU units can permanently die if a large gradient pushes them off your data manifold. In pratice, you may want to chain a few of these depending on how deep you want your deep learning to be ;) A good rule of thumb is you want just a few - maybe 1-3, unless you have really large datasets.
% 1.2 If you use SGD, you almost always want to use a non-zero momentum. 0.9 is often used for momentum. You need to play with the learning rate a bit: if it's too high, your network will never converge at best, and will die catastrophically at worst, especially if you use ReLU activations. But if it's too low, the network will take very long to train. You need to monitor the cost of the training.

%% TODO list
% 1. Max pooling
% 2. Check gradient
% 3. 