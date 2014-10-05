function [Y] = sigm(X)
%%SIGM 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 20 Sep 2014

%   Time-stamp: <2014-09-25 09:40:34 by xenuts>

    Y = 1./(1+exp(-X));
end
