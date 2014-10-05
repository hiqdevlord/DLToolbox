function [pooled_maps, pooled_indice] = maxPool3d(maps, pool_size, pool_stride, MAX_ABS)
%%MAXPOOL3D 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 20 Sep 2014

%   Time-stamp: <2014-09-20 22:28:36 by xenuts>
    
    %% Defaults
        if (nargin < 3)
        % By default, stride equals to pool size (i.e. no overlap)
            pool_stride = pool_size;    
        end
        if (nargin < 4)
        % By default, it chooses the most positive number
            MAX_ABS = 1; 
        end
    %% 3D Pooling
    % First, do 2-D pooling for all maps
    [pooled_maps, pooled_indices] = maxPool(maps, pool_size(1:2), pool_stride(1:2), MAX_ABS);

    % Pading zeros to make  mod(size(pooled_maps, 3), pool_size(3)) == 0
    padding = mod(size(pooled_maps, 3), pool_size(3));
    if (padding > 0) % need padding zeros
        padding = pool_size(3) - padding; % the dim need to be padded
        pooled_maps   = cat(3, pooled_maps, ...
                            zeros([size(pooled_maps, 1), size(pooled_maps, 2), padding, size(pooled_maps, 4)]));
        pooled_indice = cat(3, pooled_indice, ...
                            ones([size(pooled_maps, 1), size(pooled_indice, 2), padding, size(pooled_indice, 4)]));
    end
    
    % Reshape to do max pooling along the 3rd dim
    pooled_maps = reshape(pooled_maps, ...
                          [size(pooled_maps, 1), size(pooled_maps, 2), pool_size(3), ...
                        size(pooled_maps, 3)/pool_size(3), size(pooled_maps, 4)]);
    pooled_indice = reshape(pooled_indice, ...
                            [size(pooled_indice, 1), size(pooled_indice, 2),...
                   pool_size(3), size(pooled_indice,3)/pool_size(3), size(pooled_indice, 4)]);
    
    % Compute the indices over the 3rd dim
    [maxV, maxI] = max(pooled_maps, [], 3);
    if MAX_ABS
        [pooled_maps, k_indices] = max(abs(pooled_maps, [], 3));
        k_indices = k_indices - 1; %index starts from 0
        sign_inds = (maxV == pooled_maps) - 0.5; % get the sign
        pooled_maps = pooled_maps *. sign(sign_inds);
    else
        pooled_maps = maxV;
        k_indices = maxI - 1;
    end
    % Reshape back to get pooled_maps
    pooled_maps = reshape(pooled_maps, [size(pooled_maps, 1), size(pooled_maps, 2), size(pooled_maps, 4), size(pooled_maps, 5)]);
    
    inds = repmat(k_indices, [1 1 pool_size(3) 1 1]);
    sel = reshape(0:pool_size(3)-1, [1 1 pool_size(3) 1 1]);
    sel = repmat(sel, [size(inds, 1), size(inds, 2), 1, size(inds, 4), size(inds, 5)]);

    %% Has not understanding and testing yet ???
    % This should be 1 whever the pooled_indices should be taken from.
    inds = sel==inds;
    pooled_indices = pooled_indices .* inds;
    % Make the indices depend on the group, ie. 1...9 for k=1, 10..18 for k=2 if using 3x3 pooling.
    pooled_indices = pooled_indices + sel .* repmat(pool_size(1)*pool_size(2), size(sel));
    % Zero out other indices.
    pooled_indices = pooled_indices .* inds;
    % Sum over the groups (since only one in each group will be nonzero).
    pooled_indices = reshape(sum(pooled_indices, 3), size(pooled_maps));
end