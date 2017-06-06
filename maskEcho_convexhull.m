function [finalim, mask] = maskEcho_convexhull(RGBim)
grayim = removeECG(RGBim);
% imshow(grayim);
%%
bwim = im2bw(grayim, 0.001);
bwim = bwareaopen(bwim, 4000);
[x,y] = find(bwim);
conv_indices = convhull(x,y);
mask = poly2mask(y(conv_indices), x(conv_indices), size(bwim,1), size(bwim,2));
% imshow(mask);
finalim = bsxfun(@times, grayim, cast(mask,class(grayim)));
% imshow(finalim)

