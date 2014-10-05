function [Z, DropoutMask] = cnnFF(cnn, W, b, Z, PREDICT)
%%CNNFEEDFORWARD 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 20 Sep 2014

%   Time-stamp: <2014-09-25 11:53:06 by xenuts>
    if(nargin < 5)
        % Training by default, when using cnnff to predict to turn on PREDICRT
        PREDICT = 0; 
    end
    for L = 2 : numel(cnn.Layers) % except input layer
        switch cnn.Layers{L}.Type
          case 'c'
            Z{L} = cnnConv(Z{L-1}, W{L}, b{L}, cnn.Layers{L});
          case 'p'
            Z{L} = cnnPool(Z{L-1}, cnn.Layers{L});
          case {'f'}
            %% Dropout
            if(cnn.Opts.Dropout > 0)
                if (PREDICT)
                    DropoutMask{L} =[]; % WHY take "1-cnn.Opts.Dropout" somewhere (i.e. DeepLearnToolbox)
                else
                    DropoutMask{L} = (rand(cnn.Layers{L}.SizeOut, cnn.Opts.BatchSize) > cnn.Opts.Dropout);                    
                end
                Z{L} = cnnFullOut(Z{L-1}, W{L}, b{L}, cnn.Layers{L}, DropoutMask{L});
            else
                DropoutMask{L} = [];
                Z{L} = cnnFullOut(Z{L-1}, W{L}, b{L}, cnn.Layers{L});
            end
            
          case {'o'}
            DropoutMask{L} = [];
            Z{L} = cnnFullOut(Z{L-1}, W{L}, b{L}, cnn.Layers{L});
        end
    end
end