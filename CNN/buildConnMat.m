function [connMat] = buildConnMat(net, L)
%%BUILDCONNMAT 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 20 Sep 2014

%   Time-stamp: <2014-09-20 11:12:45 by xenuts>
    
    num_input  = net.Layers{L-1}.NumOut;
    num_output = net.Layers{L}.NumOut;

    switch net.Layers{L}.ConnType
      case 'full' % Full connection
        connMat = ones(num_input, num_output);
      otherwise
        error('No such connection type.');
    end
end
