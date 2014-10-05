function [W] = initWeights(cnn, L, fan_in, fan_out)
%%INITWEIGHTS 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 24 Sep 2014

%   Time-stamp: <2014-09-25 00:15:21 by xenuts>
    
    assert(strcmp(cnn.Layers{L}.Type, 'f') | strcmp(cnn.Layers{L}.Type, 'o'));
    
    if strcmp(cnn.Layers{L}.Type, 'f')
        switch cnn.Layers{L}.NonLinearity
          case 'sigmoid' % weights \in [-r r]
            r = sqrt(6 / (fan_in + fan_out));
            W = (rand(fan_in, fan_out) - 0.5) * 2 * r;
          case 'tanh'
            error('TODO: initWeights() for tanh has not implemented yet.');
          case 'relu'
            error('TODO: initWeights() for relu has not implemented yet.');
          otherwise
            error('No such NonLinearity.');
        end
    else
        switch cnn.Layers{L}.Method
          case 'softmax' % weights \in [-r r]
            r = sqrt(6 / (fan_in + fan_out));
            W = (rand(fan_in, fan_out) - 0.5) * 2 * r;
          case 'sigmoid'            
            W = ones(fan_in, fan_out);
          otherwise
            error('No such method.');
        end
    end

end