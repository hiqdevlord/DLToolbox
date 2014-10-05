function Z_u = cnnUnpool(cnn, L, Z)
%%CNNUNPOOL 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 25 Sep 2014

%   Time-stamp: <2014-09-25 22:36:48 by xenuts>

    assert(strcmp(cnn.Layers{L}.Type, 'p'), ...
           'ERROR: Unpooling operation is only applicable for pool layer.');
    
    % Pre-alloc Unpooled Z
    Z_u = zeros([cnn.Layers{L-1}.SizeOut, cnn.Layers{L-1}.NumOut, size(Z, 4)]);
    switch cnn.Layers{L}.PoolingType
      case 'Mean'
        [r, c, num_planes, num_cases] = size(Z);
        warning('TODO: Using kron() function to do mean unpooling only works when Stride == PoolingSize.');
        for i = 1 : num_cases
            for j = 1 : num_planes
                % Using kron() to do meanUnpool, but if Stride ~= PoolingSize, this method will fail.                
                pool_size = cnn.Layers{L}.PoolingSize(1:2);
                % 1/prod(pool_size) for normalizing
                Z_u(:,:,j,i) = (1 / prod(pool_size)) * kron(Z(:, :, j, i), ones(pool_size));
            end
        end
      case 'Max'
        error('TODO: max unpooling. This needs max_indices to help.');
    end    
end

