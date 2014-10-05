function norm_maps = normalizeWrapper(maps, norm_type, varargin)
%NORMALIZEWRAPPER    A wrapper to abstract out all type of normalization.
%    'local_cn': norm_maps = localcnNorm(maps, gkernel, pad_type)
%           - gkernel: a normalized filter kernel (e.g. Gaussian kernel)
%           - pad_type: specify the method to pad edge when filter. Options
%           are 'reflect' or 'zero'.
    switch norm_type
        case 'local_cn'  % local contrast normalization
            norm_maps = localcnNorm(maps, varargin{:});
        case 'whiten' % !TODO
        otherwise
            error(['Normalization type "', norm_type, '" has not implemented yet.']);
    end
end