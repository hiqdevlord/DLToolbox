function [W_g, b_g] = cnnCompGrad(cnn, L, Delta, Z)
%%CNNCOMPGRAD 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 25 Sep 2014

%   Time-stamp: <2014-09-26 00:39:48 by xenuts>
    
    assert(~strcmp(cnn.Layers{L}.Type, 'o'), ...
           'Gradient of the output layer should have already been computed outside. Plz check it.');
    assert(~strcmp(cnn.Layers{L}.Type, 'i'), 'ERROR: No gradient for input layer.');
    
    num_cases = size(Z{L}, 4);
    switch cnn.Layers{L}.Type
      case 'p' % Nothing. Currently no parameters for pooling layer
          W_g = [];
          b_g = [];
      case 'c'        
        %% W_g
        warning('TODO: computation of W_g should be replaced by valid_each3_each4() for speed-up.');
        % !!TODO: Currently, this kind of computation is extremely time-consuming, need to optimize
        for i = 1 : size(Z{L-1}, 3) %i.e. cnn.Layers{L-1}.NumOut
            for j = 1 : size(Delta{L}, 3) %i.e. cnn.Layers{L}.NumOut
                W_g_ij = zeros(cnn.Layers{L}.KernelSize);
                for k = 1 : num_cases
                    %!!TODO: valid_each3_each4
                    W_g_ij = W_g_ij + conv2(Z{L-1}(:,:,i,k), rot90(Delta{L}(:, :, j, k), 2), 'valid');
                end
                W_g(:,:,i,j) = W_g_ij / num_cases;
            end
        end
        
        %% b_g: Sum over dim 1 2 4.
        b_g = squeeze(sum(sum(sum(Delta{L}, 1), 2), 4)) ./ num_cases;
      case 'f'
        W_g = cnnConcatZ(Z{L-1}) * Delta{L}' ./ num_cases;
        b_g = sum(Delta{L}, 2) ./ num_cases;        
    end
end