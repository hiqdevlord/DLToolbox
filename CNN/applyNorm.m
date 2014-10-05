function [z] = applyNorm(Layer, z)
%%APPLYNORM 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 20 Sep 2014

%   Time-stamp: <2014-09-20 14:38:42 by xenuts>

    norm_opts = struct2cell(Layer.NormOpts);
    z = normalizeWrapper(z, Layer.NormType, norm_opts{:});    
    
end