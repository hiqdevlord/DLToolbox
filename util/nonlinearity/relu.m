function [X] = relu(X)
%%RELU 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 25 Sep 2014

%   Time-stamp: <2014-09-25 09:41:56 by xenuts>

    X = max(X, 0);
    
end