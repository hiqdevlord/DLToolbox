function [cnn, W, b, Z] = cnnInit(cnn, X, W, b, Z)
%%CNNINIT 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 19 Sep 2014

%   Time-stamp: <2014-09-25 07:48:04 by xenuts>
        
    for L = 1 : numel(cnn.Layers)
        switch cnn.Layers{L}.Type
          case 'i' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [r, c, num_planes, num_cases] = size(X);
            num_input = num_planes;
            input_size = [r, c];
            cnn.Layers{L}.SizeOut = input_size;
            cnn.Layers{L}.NumOut = num_input;
            cnn.Opts.NumCases = num_cases;
            cnn.Opts.NumBatches = num_cases / cnn.Opts.BatchSize;
            
            %% Normalization            
            X = applyNorm(cnn.Layers{L}, X); %% !!!!!! Temporary Comment Out
            W{L} = []; b{L} = []; Z{L} = X;
            clear X;
          case 'c' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            input_size = input_size - cnn.Layers{L}.KernelSize + 1;
            cnn.Layers{L}.SizeOut = input_size;
            num_input = cnn.Layers{L}.NumOut;
            
            cnn.Layers{L}.ConnMat = buildConnMat(cnn, L); % create connectivity matrix
            W{L} = initFilter(cnn, L); % initialize filters
            b{L} = zeros(cnn.Layers{L}.NumOut, 1);
            Z{L} = []; % zeros([cnn.Layers{L}.SizeOut, cnn.Layers{L}.NumOut, cnn.Opts.NumCases]);
          case 'p' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            input_size = ...
                ceil((input_size - cnn.Layers{L}.PoolingSize(1:2)) ./ cnn.Layers{L}.Stride(1:2)) + 1;
            cnn.Layers{L}.SizeOut = input_size;
            %num_input =  ceil((cnn.Layers{L-1}.NumOut - cnn.Layers{L}.PoolingSize(3)) ./ cnn.Layers{L}.Stride(3)) + 1;
            cnn.Layers{L}.NumOut = num_input;
            W{L} = []; b{L} = []; Z{L} = []; 
          case {'f', 'o'}  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if (size(input_size, 2) == 1) % this layer is NOT the first fully connected layer.
                if (strcmp(cnn.Layers{L}.SizeOut, 'same'))
                    cnn.Layers{L}.SizeOut = input_size;
                else % a numeric figure
                    input_size = cnn.Layers{L}.SizeOut;
                end
                fan_in = cnn.Layers{L-1}. SizeOut;
                fan_out = cnn.Layers{L}.SizeOut;
            elseif(size(input_size, 2) == 2) % this layer is the first fully connected layer.
                input_size = prod(input_size) * num_input; % Become a scalar from now on
                clear num_input; % No more inferring                
                fan_in = input_size;
                if (strcmp(cnn.Layers{L}.SizeOut, 'same'))
                    cnn.Layers{L}.SizeOut = input_size;
                else % a numeric figure
                    input_size = cnn.Layers{L}.SizeOut;
                end
                fan_out = cnn.Layers{L}.SizeOut;
            else
                error('ERROR: Bad size of input_size!');
            end
            W{L} = initWeights(cnn, L, fan_in, fan_out);
            b{L} = zeros(cnn.Layers{L}.SizeOut, 1);
            Z{L} = [];
          otherwise
            error(['No such a layer type of ', cnn.Layers{L}.type]);
        end
    end
end