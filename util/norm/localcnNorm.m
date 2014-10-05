function norm_maps = localcnNorm(maps, varargin)
%LOCALCNNORM    2-D local contrast normalization.
    %% Defaults
    if(nargin < 2)
        pad_type = 'reflect'; % Options: 'reflect', 'zero'
    else
        pad_type = varargin{2};
    end
    if(nargin < 1)
        [r, c, ~] = size(map);
        k = fspecial('gaussian', [round(sqrt(r)/2) round(sqrt(c)/2)], 5);  
    else
        k = varargin{1};
    end
    if(nargin < 1)
        error('Error: localContrastNormalization() requires at least 1 parameter.');
    end

    %%
    map_size = size(maps);
    [xdim, ydim, dimprod] = size(maps);
    
    [kr, kc] = size(k);
    
    norm_maps = zeros([xdim, ydim, dimprod]);
    
    for c = 1 : dimprod
        Ic = double(maps(:,:,c));
        r0 = floor( (kr-1)/2 );
        c0 = floor( (kc-1)/2 );
        switch pad_type
            case 'reflect' % Padding with reflected copies
                Ic0 = [ Ic(kr-r0:-1:2,kc-c0:-1:2),        Ic(kr-r0:-1:2,:),        Ic(kr-r0:-1:2,ydim-1:-1:ydim-c0); ...
                        Ic(:,kc-c0:-1:2),                 Ic,                      Ic(:,ydim-1:-1:ydim-c0); ...
                        Ic(xdim-1:-1:xdim-r0,kc-c0:-1:2), Ic(xdim-1:-1:xdim-r0,:), Ic(xdim-1:-1:xdim-r0,ydim-1:-1:ydim-c0) ];
            case 'zero' % Padding with zeros
                Ic0 = [ zeros(kr-r0-1, ydim+kc-1);
                        zeros(xdim, kc-c0-1), Ic, zeros(xdim, c0);
                        zeros(r0, ydim+kc-1); ];
            otherwise 
                error(['No such a padding type "' pad_type '".']);
        end
        %% Subtractive Normalization: i.e. zero mean
        % v_{i,j} = x_{i,j} - \sum_{pq}(w_{p,q} \times x_{i+p,j+q}),
        % w_{p,q} is a Gaussian weighting window so that \sum{pq}w_{p,q}=1
        lmn = conv2(Ic0, k, 'valid'); % weighted mean
        Is = Ic - lmn; % substact the mean
        
        %% Divisive Normalization: i.e. unit variance
        % x_{i,j} = v_{i,j} / max(c, \delta_{i,j})
        % \delta_{i,j} = (\sum{i,j}(w_{p,q} \times v_{i,j}^2 ))^{1/2}, and
        % the constant c is usually set the mean(\delta_{i,j})
        % i.e. each point divides the (Gaussian) weighted standard deviation (std) of its neighborhood
        % Normally: Variance D(X) = E([X - E(X)]^2) = E(X^2) - (E(X))^2
        % std = sqrt(D(X))
        % Here: a weighted version
        lmnsq = conv2(Ic0.^2, k, 'valid'); % E(X^2)
        lvar = lmnsq - lmn.^2; % D(X) = E(X^2) - (E(X)^2)
        lstd = sqrt(lvar); % sqrt(D(X))

        th = mean(lstd(:));

        lstd(lstd <= th) = th;
        lstd(lstd(:) == 0) = eps;
        
        Icd = Is ./ lstd;
        
        %%
        norm_maps(:, :, c) = Icd;
    end
    
    %% Reshape
    norm_maps = reshape(norm_maps, [xdim, ydim, map_size(3:end)]);
end
