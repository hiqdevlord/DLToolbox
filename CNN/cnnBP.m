function [Cost, Grad] = cnnBP(cnn, W, b, Z, Y, DropoutMask, pooled_ind)
%%CNNBP 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 25 Sep 2014

%   Time-stamp: <2014-10-06 08:19:21 by xenuts>
    
    
    %% Cost Function
    Delta = cell(1, numel(cnn.Layers)); % Error Sensitivity
    Grad = cell(1, numel(cnn.Layers)); % Gradients of W and b
    
    %% Regularization Cost
    reg_cost = 0;
    for L = 2 : numel(cnn.Layers)
        reg_cost = reg_cost + sum(W{L}(:) .^ 2);
    end
    
    %% Compute cost of the last layer, i.e. output layer
    switch cnn.Layers{end}.Method
      case 'softmax' % cross-entropy cost for Softmax is the most practice choice
        num_cases = numel(Y);
        %% Groundtruth: one-hot fashion
        T = full(sparse(Y, 1:num_cases, 1));
    %     assert(size(T, 1) == cnn.Layers{end}.SizeOut, 'Try again may help or you need check output layer size.');
        if(size(T, 1) < cnn.Layers{end}.SizeOut)
            T = [T; zeros(cnn.Layers{end}.SizeOut - size(T,1), num_cases)];
        end
    
        probs = Z{end}; % Z{end} is exactly the prob of softmax
        
        %% Cost = Cross-Entropy Loss + Regularization
       
        Cost = -1./num_cases * T(:)' * log(probs(:)) + 0.5 * cnn.Opts.Lambda .* reg_cost; % see 'prml.pdf'
        % Equal => -1./num_cases * sum(sum(T .* log(probs), 1), 2) + 1/2 * ||W||^2
        
        Delta{end} = probs - T; % When using cross-entropy loss, the delta is 
        % If use MSE loss, plz See details in http://www.cnblogs.com/tornadomeet/p/3468450.html
        
        % Grad{end}{1} and Grad{end}{2} store gradients of W and b.        
        Grad{end}{1} = (1./num_cases) * cnnConcatZ(Z{end-1}) * Delta{end}' + cnn.Opts.Lambda .* W{end}; %Grad W
        Grad{end}{2} = (1./num_cases) * sum(Delta{end}, 2); % Grad b
      case 'sigmoid'% Loss Function is MSE
        num_cases = size(Y, 1);
        error('XTodo: cost function for sigmoid has NOT been implemented yet !');
        Cost = -1./num_cases * sum((Z{end} - Y').^2, 2) + 0.5 * cnn.Opts.Lambda .* reg_cost;
      case 'linear'
        error('XTodo: cost function for linear has NOT been implemented yet !');
    end    
    
    for L = numel(cnn.Layers)-1 : -1 : 2 %except input and output layers
        %% Compute Delta
        Delta = cnnCompDelta(cnn, L, Delta, W, b, Z, DropoutMask, pooled_ind);
        
        %% Compute Grad        
        [Grad{L}{1}, Grad{L}{2}] = cnnCompGrad(cnn, L, Delta, W, Z);       
    end    
end