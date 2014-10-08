function [pooled_maps] = meanPool(maps, pool_size, pool_stride)
%%MEANPOOL 2D mean pooling operation
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 25 Sep 2014

%   Time-stamp: <2014-10-06 08:38:02 by xenuts>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Defaults
    if (nargin < 3)
    % By default, stride equals to pool size (i.e. no overlap)
        pool_stride = pool_size;    
    end
    if (nargin < 4)
    % By default, it chooses the most positive number
        MAX_ABS = 1; 
    end
    
    %%
    map_size = size(maps);
    [rdim, cdim, dimprod] = size(maps);
    
    % The size of pooled_maps (also pooled_indices)
    pooled_size = ceil( ([rdim, cdim] - pool_size(1:2)) ./ pool_stride(1:2) ) + 1;

    % A pooling block(or region)
    block_elem_num = pool_size(1) * pool_size(2);
    block_map = zeros( [block_elem_num, dimprod] );

    rIs = 1 : pool_size(1);
    cIs = 1 : pool_size(2);

    % Initialize
    pooled_maps = zeros([ pooled_size, dimprod ]);

    %% Mean pooling operation
    for ii = 0 : pooled_size(1)-1
        for jj = 0 : pooled_size(2)-1
            % Get current block. Here using ':' operator to concatenate all the other dims
            block_map(1:block_elem_num, :) = reshape( maps( ...
                min(ii*pool_stride(1)+rIs, rdim), ... %% row range
                min(jj*pool_stride(2)+cIs, cdim), :), ... %% col range
                                                      block_elem_num, dimprod );             

            % Get Mean            
            pooled_maps(ii+1, jj+1, :) = mean(block_map, 1);% along the 1st dim
        end
    end
    
    %% Reshape
    pooled_maps = reshape(pooled_maps, [pooled_size, map_size(3:end)]);
    
end