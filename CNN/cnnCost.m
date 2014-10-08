function [Cost, Grad] = cnnCost(cnn, W, b, Z_batch, Y_batch)
%%CNNCOST 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 07 Oct 2014

%   Time-stamp: <2014-10-07 09:27:54 by xenuts>

    % Feed-Forward
    [Z_batch, DropoutMask, pooled_ind] = cnnFF(cnn, W, b, Z_batch);
    % Back-Propagation
    [Cost, Grad] = cnnBP(cnn, W, b, Z_batch, Y_batch, DropoutMask, pooled_ind);
    
end