function [Z, pooled_ind] = cnnPool(Z, Layer)
%%APPLYPOOLING 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 20 Sep 2014

%   Time-stamp: <2014-10-06 08:08:22 by xenuts>
    
    [Z, pooled_ind] = poolWrapper(Z, Layer.PoolingType, Layer.PoolingSize, Layer.Stride, Layer.MAX_ABS);
    
end