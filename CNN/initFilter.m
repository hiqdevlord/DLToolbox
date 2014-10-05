function [F] = initFilter(net, L)
%%INITFILTER 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 20 Sep 2014

%   Time-stamp: <2014-09-25 08:05:50 by xenuts>
    if(strcmp(net.Layers{L}.Type, 'c'))
        F = randn([net.Layers{L}.KernelSize, net.Layers{L-1}.NumOut, net.Layers{L}.NumOut]);
    else % Other type of layers has no filters
        warning(['WARNING: No filters in Layer #', num2str(L)]);
        F = [];
    end

end
