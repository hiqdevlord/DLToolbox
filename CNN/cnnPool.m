function [Z] = cnnPool(Z, Layer)
%%APPLYPOOLING 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 20 Sep 2014

%   Time-stamp: <2014-09-25 10:24:24 by xenuts>
    
    [Z, ~] = poolWrapper(Z, Layer.PoolingType, Layer.PoolingSize, Layer.Stride, Layer.MAX_ABS);
    
end