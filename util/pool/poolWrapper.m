function [pooled_maps, pooled_indice] = poolWrapper(maps, pool_type, pool_size, pool_stride, MAX_ABS)
    pooled_indice =[];
    switch pool_type
      case 'Max' %2-D
        [pooled_maps, pooled_indice] = maxPool(maps, pool_size, pool_stride, MAX_ABS);
      case 'Max3' %3-D
        [pooled_maps, pooled_indice] = maxPool3d(maps, pool_size, pool_stride, MAX_ABS);
      case 'Mean' 
        pooled_maps = meanPool(maps, pool_size, pool_stride); % !TODO
      case 'Mean3'
        [pooled_maps, pooled_indice] = meanPool3d(); % !TODO
      otherwise
        error(['No such pooling type: "', pool_type, '".']);
    end
end
