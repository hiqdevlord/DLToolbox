function [Z] = cnnConcatZ(Z)
%%CNNCONCATZ 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 25 Sep 2014

%   Time-stamp: <2014-09-25 16:18:11 by xenuts>

    assert(numel(size(Z))==4 | (numel(size(Z))==2 & numel(Z) > 1));
    
    if (numel(size(Z)) == 4)
        [r, c, num_planes, num_cases] = size(Z);
        % Concatenate
        Z = reshape(Z, [r*c*num_planes, num_cases]);
    else
        Z = Z; % Do nothing
    end
    
end