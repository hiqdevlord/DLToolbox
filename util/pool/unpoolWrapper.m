function [unpooled_maps] = unpoolWrapper(maps, pool_type, pool_size, pool_stride, pooled_ind)
%%UNPOOLWRAPPER 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 06 Oct 2014

%   Time-stamp: <2014-10-06 09:24:26 by xenuts>

    if(nargin < 5)
        pooled_ind = [];
    end
    
    switch pool_type
      case 'Max' 
        assert(~isempty(pooled_ind), 'Error: pooled_ind is required when maxUnpool!!!')
        assert(all(size(maps) == size(pooled_ind)), 'Error: Sizes of maps and pooled_ind shoule be the same!');
        unpooled_maps = maxUnpool(maps, pool_size, pool_stride, pooled_ind);
      case 'Max3' 
        error('TODO: Max3 unpool has not been implemented yet !');
      case 'Mean'
        error('TODO: Mean unpool has not been implemented yet !');
      case 'Mean3'
        error('TODO: Mean3 unpool has not been implemented yet !');
      otherwise
        error(['No such pooling type: "', pool_type, '".']);
    end

end