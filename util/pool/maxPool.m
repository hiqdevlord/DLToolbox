function [pooled_maps, pooled_indice] = maxPool(maps, pool_size, pool_stride, MAX_ABS)
%MAXPOOL 2D max pooling operation.
%   MAXPOOL(MAPS, POOL_SIZE, POOL_STRIDE, MAX_ABS) returns the pooled maps.
%   Inputs:
%      - MAPS: the input data, [XDIM x YDIM X ...].
%      - POOL_SIZE: the size of pooling operation, [SIZE1 x SIZE2 x ...].
%      - POOL_STRIDE: the stripe when pooling moves on. It is POOL_SIZE by
%      default.
%      - MAX_ABS: whether choose the element with maximum absolute value or
%      simply the most positive element. It is 0 by default.
%   Outputs:
%      - POOLED_MAPS: max pooled maps.
%      - POOLED_INDICE: the position of selected maximum value. Starts at 0. 
%   Size of POOLED_MAPS: 
%      ceil( ([XDIM, YDIM] - POOL_SIZE(1:2)) / POOL_STRIDE(1:2) ) + 1

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
pooled_indice = zeros([ pooled_size, dimprod ]);

%% Max pooling operation
for ii = 0 : pooled_size(1)-1
    for jj = 0 : pooled_size(2)-1
       % Get current block. Here using ':' operator to concatenate all the other dims
       block_map(1:block_elem_num, :) = reshape( maps( ...
           min(ii*pool_stride(1)+rIs, rdim), ... %% row range
           min(jj*pool_stride(2)+cIs, cdim), :), ... %% col range
           block_elem_num, dimprod ); % Note: here using : to concatenate all the other dims

       % Find the maximum
       [maxV, maxI] = max(block_map, [], 1);
       maxI = maxI-1; % Start at 0.
       if MAX_ABS % Find the one with maximum absolute value
           [minV, minI] = min(block_map, [], 1);
           minI = minI-1; % Start at 0.
           maxes = minV;
           gtI = ( maxV >= abs(minV));
           maxes(gtI) = maxV(gtI);
           minI(gtI) = maxI(gtI);
           pooled_maps(ii+1, jj+1, :) = maxes;
           pooled_indice(ii+1,jj+1,:) = minI;               
       else % Find the most positive one
           pooled_maps(ii+1, jj+1, :) = maxV;
           pooled_indice(ii+1,jj+1,:) = maxI;
       end
   end
end

%% Reshape
pooled_maps = reshape(pooled_maps, [pooled_size, map_size(3:end)]);
pooled_indice = reshape(pooled_indice, [pooled_size, map_size(3:end)]);

end % end of function