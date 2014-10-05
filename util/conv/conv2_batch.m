function out = conv2_batch(maps, F, C, which_dim_sum, shape)
%CONV2_BATCH 2D convolution on a batch of image data.
%   CONV2_BATCH(MAPS, F, C, WHICH_DIM_SUM, SHAPE)
%   Inputs:
%   - MAPS: A batch of image data is stored in MAPS. Typically, MAPS's shape is
%           r_in x c_in x num_planes_in x num_image_cases. [r, c] is the size of each image plane.
%           num_planes typically specifies the numbers of color channel (i.e. 1 for gray 
%           and 3 for RGB) or more generally the numbers of feature map (i.e. in the 
%           context of convolutional neural network). num_image_cases is the number of 
%           total image cases.
%   - F: The filters to convolve with MAPS. It size is p x q x num_planes1 x num_planes2 [x num_image_cases].
%        [p, q] is the size of filters. num_planes1 and num_planes2 are the numbers of planes 
%        before and after convolution. If WHICH_DIM_SUM=3, then num_planes1=num_planes_in, num_planes2=num_planes_out;
%        otherwise if WHICH_DIM_SUM=4, then num_planes2=num_planes_in, num_planes1=num_planes_out.
%        When num_image_cases is missed, it equals to 1 by default which means filter banks are shared by all image cases.
%        Otherwise, each image case has a distinct bank of filters.
%   - C: A num_planes1 x num_planes2 binary matrix to specify the connectivity between planes
%        before and after convolution.
%   - WHICH_DIM_SUM: It specifies which dimension of F to be summed up. Its value is 3 or 4.
%   - SHAPE: 'full' or 'valid'
%   Outputs:
%   - out: convoluted results, its shape is r_out x c_out x num_planes_out x num_image_cases.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(size(C,1) ~= size(F,3))
    error('Connectivity matrix first dimension does not match second inputs third dimension.')
end
if(size(C,2) ~= size(F,4))
    error('Connectivity matrix second dimension does not match second inputs fourth dimension.')
end
if(which_dim_sum==4 && size(C,2) ~= size(maps,3))
    error('Connectivity matrix second dimension does not match first inputs third dimension.')
elseif(which_dim_sum==3 && size(C,1) ~= size(maps,3))
    error('Connectivity matrix first dimension does not match first inputs third dimension.')
end

%% Initialize the out according to shape and which_dim_sum
switch shape
  case 'valid'
    out = zeros(size(maps,1)-size(F,1)+1, size(maps,2)-size(F,2)+1,...
                size(F, 7-which_dim_sum), size(maps,4));
  case 'full'
    out = zeros(size(maps,1)+size(F,1)-1, size(maps,2)+size(F,2)-1, ...
                size(F, 7-which_dim_sum), size(maps,4));
  case 'same'
    error(['shape ', shape, 'is not implemented currently!' ]);
  otherwise
    error(['No such shape ', shape, 'is supported!' ]);
end

%% Do convolution over each image case
for im = 1 : size(maps, 4)
   for j = 1 : size(F, 3) %num_planes1
      for k = 1 : size(F, 4) %num_planes2
          if (C(j, k) ~= 0) % Only convolve on existing connection
              switch which_dim_sum
                case 3 % num_planes1 == num_planes_in, num_planes2 == num_planes_out
                  if(size(F, 5) > 1) % Each image cases has a distinct bank of filters
                      out(:,:,k,im) = out(:,:,k,im) + C(j,k)*conv2(maps(:,:,j,im),F(:,:,j,k,im),shape);
                  else % All image cases share a bank of filters
                      out(:,:,k,im) = out(:,:,k,im) + C(j,k)*conv2(maps(:,:,j,im),F(:,:,j,k),shape);
                  end
                case 4 % num_planes2 == num_planes_in, num_planes1 == num_planes_out.
                  if(size(F, 5) > 1)
                      out(:,:,j,im) = out(:,:,j,im) + C(j,k)*conv2(maps(:,:,k,im),F(:,:,j,k,im), shape);
                  else
                      out(:,:,j,im) = out(:,:,j,im) + C(j,k)*conv2(maps(:,:,k,im),F(:,:,j,k), shape);
                  end
              end
          end
      end % k       
   end % j
end % im