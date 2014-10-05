function Z = cnnConv(Z, F, b, Layer)
%%CNNCONV 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 25 Sep 2014

%   Time-stamp: <2014-09-25 09:44:11 by xenuts>
    
    %% Rotate Filters
    [~, ~, num_in, num_out] = size(F);
    for i = 1 : num_in
        for j = 1 : num_out
            % Rotate the filters 180 degree, because in CNN farward pass, here is actually a 
            % correlation operation
            F(:, :, i, j) = rot90(F(:, :, i, j), 2);
        end
    end
    
    %% Convolve
    % Conv in batch fashion (i.e. 4-D matrix with shape of [r x c x num_planes x num_cases])
    Z = conv2_batch(Z, F, Layer.ConnMat, 3, 'valid');
    
    %% Add bias and apply non-linearity
    [~, ~, num_planes, num_cases] = size(Z);
    assert(num_planes == numel(b), 'Dimensions of Z_conv and b are mis-matched !');
    for i = 1 : num_cases
        for j = 1 : num_planes
            Z(:, :, j, i) = Z(:, :, j, i) + b(j); % one plane shares same bias
        end
    end    
    
    switch Layer.NonLinearity
      case 'sigmoid'
        Z = sigm(Z);
      case 'tanh'
        Z = tanh(Z);
      case 'relu'
        Z = relu(Z);
    end
end