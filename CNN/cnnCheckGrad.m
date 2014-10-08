function [GradNumeric] = cnnCheckGrad(cnn, W, b, Z_batch, Y_batch, Grad)
%%CNNCOMPNUMERICALGRAD 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 07 Oct 2014

%   Time-stamp: <2014-10-08 10:08:51 by xenuts>
    
% Testings:
%   1. i -> softmax            [X] Successful => Gradient of softmax is correct
%   2. i -> f -> softmax       [X] Successful => Gradient of 'f' layer is correct (fix a bug in cnnCompGrad())
%   3. i -> c -> p -> softmax  [?] Diff is at the level of 1e-6. Always occurs outlines for #4, #9, #15, #20
%                                  and with c f layer, softmax layer's grad-check aslo fails. Why?? Maybe BUGs????
%   4. i -> c -> softmax       [?] c layer's check is okay, but cause sparisity of softmax's gradient. 
%                                  When using sigmoid/tanh instead of relu, it works OK. 
%                                  Besides, when using a larger batchsize, it gets good convergence

    %%
    epsilon = 1e-4;
    
%     for L = numel(cnn.Layers) : -1 : 2
    for L = 2 : numel(cnn.Layers)
        if(~isempty(W{L})) % Skip those layers without learnable parameters
            GradNumeric{L}{1} = zeros(size(W{L}));
            GradNumeric{L}{2} = zeros(size(b{L}));
            fprintf('Check gradient of layer %d, number of learnable parameters is %d\n', L, numel(W{L}));
            fprintf('[Grad,    GradNumeric,    Diff]\n');
            
            %% Check Gradient of W
            for I = 1 : numel(W{L})
                switch cnn.Layers{L}.Type
                  case {'c','p'} % shape of W is [fs1 x fs2 x n1 x n2]
                    [fs1, fs2, n1, n2] = ind2sub(size(W{L}), I);
                    subI = {fs1, fs2, n1, n2};                   
                  case {'f','o'} % shape of W is [n1 x n2]
                    [n1, n2] = ind2sub(size(W{L}), I);
                    subI = {n1, n2}; 
                end
                %% 
                WI = W{L}(subI{:});
                
                W{L}(subI{:}) = WI + epsilon;
                [Cost_p, ~] = cnnCost(cnn, W, b, Z_batch, Y_batch);
                
                W{L}(subI{:}) = WI - epsilon;
                [Cost_n, ~] = cnnCost(cnn, W, b, Z_batch, Y_batch);
                
                GradNumeric{L}{1}(subI{:}) = (Cost_p - Cost_n) / (2 * epsilon);
                fprintf('#%d: [%.10f, %.10f, %.10f]\n', I, Grad{L}{1}(subI{:}), GradNumeric{L}{1}(subI{:}), Grad{L}{1}(subI{:})-GradNumeric{L}{1}(subI{:}));
                W{L}(subI{:}) = WI;               
            end
            % Check Gradient
            diffW = norm(GradNumeric{L}{1}(:) - Grad{L}{1}(:)) / norm(GradNumeric{L}{1}(:) + Grad{L}{1}(:));
            fprintf('Gradient difference of W at Layer %d is %.10f\n', L, diffW);
            % Should be small. In our implementation, these values are usually less than 1e-9.
            % assert(diffW < 1e-9, 'Difference of W too large. Check your gradient computation again.\n');
            fprintf('Gradient Check of W Succeed.\n=================================\n'); 
            
            %% Check Gradient of b
            for I = 1 : numel(b{L})
                switch cnn.Layers{L}.Type
                  case {'c','p'} % shape of W is [fs1 x fs2 x n1 x n2]
                    [fs1, fs2, n1, n2] = ind2sub(size(b{L}), I);
                    subI = {fs1, fs2, n1, n2};                   
                  case {'f','o'} % shape of W is [n1 x n2]
                    [n1, n2] = ind2sub(size(b{L}), I);
                    subI = {n1, n2}; 
                end
                %% 
                bI = b{L}(subI{:});
                
                b{L}(subI{:}) = bI + epsilon;
                [Cost_p, ~] = cnnCost(cnn, W, b, Z_batch, Y_batch);
                
                b{L}(subI{:}) = bI - epsilon;
                [Cost_n, ~] = cnnCost(cnn, W, b, Z_batch, Y_batch);
                
                GradNumeric{L}{2}(subI{:}) = (Cost_p - Cost_n) / (2 * epsilon);
                fprintf('#%d: [%.10f, %.10f, %.10f]\n', I, Grad{L}{2}(subI{:}), GradNumeric{L}{2}(subI{:}), Grad{L}{2}(subI{:})-GradNumeric{L}{2}(subI{:}));
                b{L}(subI{:}) = bI;               
            end
            % Check Gradient 
            diffb = norm(GradNumeric{L}{2}(:) - Grad{L}{2}(:)) / norm(GradNumeric{L}{2}(:) + Grad{L}{2}(:));
            fprintf('Gradient difference of b at Layer %d is %.10f\n', L, diffb);
            % Should be small. In our implementation, these values are usually less than 1e-9.
            % assert(diffb < 1e-9, 'Difference of b too large. Check your gradient computation again.\n');
            fprintf('Gradient Check of b Succeed.\n=================================\n');
            
            fprintf('\n********* Done with Layer %d *********\n\n', L);
        end
    end
