function [W, b, Ln] = cnnLearn(cnn, W, b, Grad, Ln)
%%CNNLEARNSGD 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 26 Sep 2014

%   Time-stamp: <2014-09-26 01:22:48 by xenuts>
    for L = 2 : numel(cnn.Layers)
        if(~isempty(Grad{L}{1})) % skip over these void cell in Grad
            if(numel(Ln.velocity) < L) % First time
                Ln.velocity{L}{1} = zeros(size(Grad{L}{1}));
                Ln.velocity{L}{2} = zeros(size(Grad{L}{2}));
            end
            %%Qs: Why using mom leads to unconvergent ???
            %% Updating W and b
            if(cnn.Opts.Momentum > 0)
                Ln.velocity{L}{1} = Ln.mom .* Ln.velocity{L}{1} + Ln.alpha .* Grad{L}{1}; %W
                Ln.velocity{L}{2} = Ln.mom .* Ln.velocity{L}{2} + Ln.alpha .* Grad{L}{2}; %b
            else
                Ln.velocity{L}{1} = Ln.alpha .* Grad{L}{1}; %W
                Ln.velocity{L}{2} = Ln.alpha .* Grad{L}{2}; %b
            end
            
            W{L} = W{L} - Ln.velocity{L}{1};
            b{L} = b{L} - Ln.velocity{L}{2};
        end
    end
end