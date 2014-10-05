function Delta = cnnCompDelta(cnn, L, Delta, W, b, Z, DropoutMask)
%%CNNCOMPDELTA 
%   
%   Copyright (C) 2014 by Xiangzeng Zhou
%   Author: Xiangzeng Zhou <xenuts@gmail.com>
%   Created: 25 Sep 2014

%   Time-stamp: <2014-09-25 22:40:38 by xenuts>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    switch cnn.Layers{L}.Type
      case 'f' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Currently allowed patterns: 'f'->'f' or 'f'->'o'
        switch cnn.Layers{L+1}.Type
          case {'f', 'o'}
            dFunc = cnnGetDerivateFunc(cnn.Layers{L});
            Delta{L} = W{L+1} * Delta{L+1} .* dFunc(Z{L});
            if(~isempty(DropoutMask{L}))
                Delta{L} = Delta{L} .* DropoutMask{L};
            end
          otherwise
            error('This pattern currently has NOT been implemented.');
        end
      case 'p' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Currently allowed patterns: 'p'->'f' or 'p'->'o' or 'p'->'c'
        dFunc = cnnGetDerivateFunc(cnn.Layers{L});
        switch cnn.Layers{L+1}.Type
          case {'f', 'o'}            
            Delta{L} = W{L+1} * Delta{L+1} .* dFunc(Z{L});
            % From here on, the delta is reshaped.
            Delta{L} = reshape(Delta{L}, [cnn.Layers{L}.SizeOut, cnn.Layers{L}.NumOut, size(Delta{L}, 2)]);
          case 'c'
            % Here it's a real conv operation, no need rotating.
            Delta{L} = conv2_batch(Delta{L+1}, W{L+1}, cnn.Layers{L+1}.ConnMat, 4, 'full') .* dFunc(Z{L});
          otherwise
            error('This pattern currently has NOT been implemented.');
        end
      case 'c' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      % Currently allowed patterns: 'c'->'p' or 'c'->'c'
        dFunc = cnnGetDerivateFunc(cnn.Layers{L});
        switch cnn.Layers{L+1}.Type
          case 'p'
            Delta{L} = cnnUnpool(cnn, L+1, Delta{L+1}) .* dFunc(Z{L});
          case 'c'
            % Here it's a real conv operation, no need rotating.
            Delta{L} = conv2_batch(Delta{L+1}, W{L+1}, cnn.Layers{L+1}.ConnMat, 4, 'full') .* dFunc(Z{L});
          case {'f'}
            Delta{L} = W{L+1} * Delta{L+1};
            Delta{L} = reshape(Delta{L}, [cnn.Layers{L}.SizeOut, cnn.Layers{L}.NumOut, size(Delta{L}, 2)]);
            Delta{L} = Delta{L} .* dFunc(Z{L});            
          otherwise
            error('This pattern currently has NOT been implemented.');
        end
      otherwise
        error('ERROR: Only back propagate over layer f p c.');
    end
end
