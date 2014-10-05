function [dFunc] = cnnGetDerivateFunc(Layer)
%%CNNGETDERIVATEFUNC 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 25 Sep 2014

%   Time-stamp: <2014-09-25 21:48:40 by xenuts>

    assert(strcmp(Layer.Type, 'c') | strcmp(Layer.Type, 'f') | strcmp(Layer.Type, 'p'),  ...
           'ERROR: Currently only conv, pooling and full layers have NonLinearity.');
    if strcmp(Layer.Type, 'p')
         % Currently, we don't consider non-linearity for pooling layer.
         % So here we set a constant 1 function as a place holder for future extension.
        dFunc = @(x)1;
    else % for 'c' and 'f'
        switch Layer.NonLinearity
          case 'sigmoid'
            dFunc = @(fx)fx.*(1-fx); %sigm(x)' = sigm(x)*(1-sigm(x));
          case 'relu'
            dFunc = @(fx)double(fx > 0); %relu(x)' = (relu(x) > 0)
          case 'tanh'
            dFunc = @(fx)1 - fx.^2; %tanh(x)' = 1 - tanh(x)^2
        end    
    end
end
