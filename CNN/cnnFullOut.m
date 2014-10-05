function Z = cnnFullOut(Z, W, b, Layer, DropoutMask)
%%CNNFULLOUT
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 25 Sep 2014

%   Time-stamp: <2014-09-25 20:57:18 by xenuts>
    if(nargin < 5)
        DropoutMask = [];
    end
    Z = cnnConcatZ(Z);
    
    % Add bias
    Z = W' * Z;
    Z = Z + repmat(b, [1, size(Z, 2)]);
    
    if strcmp(Layer.Type, 'f') %%% Fully connected layer
        switch Layer.NonLinearity
          case 'sigmoid'
            Z = sigm(Z);
          case 'tanh'
            Z = tanh(Z);
          case 'relu'
            Z = relu(Z);
        end
        if(~isempty(DropoutMask))
            Z = Z .* DropoutMask;
        end
    else %%%%%%%%%% Output layer
        switch Layer.Method
          case 'sigmoid'
            Z = sigm(Z);
          case 'linear'
            Z = Z;
          case 'softmax'
            % subtract out the maximum to prevent overflow
            M = bsxfun(@minus, Z, max(Z, [], 1));
            M = exp(M);
            Z = bsxfun(@rdivide, M, sum(M, 1));
% % %             if(~isempty(find(Z(:) == 0)))
% % %                 disp(1);
% % %             end
        end
    end
end