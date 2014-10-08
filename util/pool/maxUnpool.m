function [unpooled_maps] = maxUnpool(maps, pool_size, pool_stride, pooled_ind)
%%MAXUNPOOL 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 06 Oct 2014

%   Time-stamp: <2014-10-06 10:17:51 by xenuts>
    
    [Rp, Cp, num_planes, num_cases] = size(maps);

    Ru = (Rp - 1) .* pool_stride(1) + pool_size(1);
    Cu = (Cp - 1) .* pool_stride(2) + pool_size(2);
    unpooled_maps = zeros(Ru, Cu, num_planes, num_cases);
    
    rid = 1 : pool_size(1);
    cid = 1 : pool_size(2);
    for r = 0 : Rp - 1
        for c = 0 : Cp - 1
            %% Get a bunch of pooled_ind at the position (r+1, c+1)
            pooled_ind_block = reshape(squeeze(pooled_ind(r+1, c+1, :, :)), [1, num_planes*num_cases]) + 1; % start from 1
            % One-hot scheme
            pooled_ind_block = full(sparse(pooled_ind_block, 1:numel(pooled_ind_block), 1));
            % pad zeros to match the length of pool_size(1) * pool_size(2)
            pooled_ind_block = [pooled_ind_block; zeros(prod(pool_size(1:2))-size(pooled_ind_block, 1), size(pooled_ind_block, 2))];
            
            %% Get a bunch of pooled maps at the position (r+1, c+1)
            maps_block = reshape(squeeze(maps(r+1, c+1, :, :)), [1, num_planes*num_cases]);
            % repmat to match the size of pooled_ind_block
            maps_block = repmat(maps_block, [prod(pool_size(1:2)), 1]);

            %% Element-wise producting and reshape to pool_size
            unpooled_maps_block = reshape(pooled_ind_block .* maps_block, ...
                                          [pool_size(1), pool_size(2), num_planes, num_cases]);
            
            %% Copy to corresponding block in unpooled_maps
            unpooled_maps(min(r*pool_stride(1)+rid, Ru), ...
                          min(c*pool_stride(2)+cid, Cu), :, :) = unpooled_maps_block;
        end
    end

end