function im = mask_and_crop_echo(im, W, H)

meanImageW_orig = 500; meanImageH_orig = 370;
startCropX_orig = 68; startCropY_orig = 82;
scaleX = W/400;
scaleY = H/267;

echoMask = imread('masks/manualMask.bmp');
echoMask(echoMask>1) = 1;
echoMask = imresize(echoMask, size(echoMask).*[scaleY,scaleX]);

if size(im, 3) > 1
    im = rgb2gray(im);
end
im = removeECG(im);

im = imresize(im, [meanImageH_orig*scaleY,meanImageW_orig*scaleX]);                        
im = imcrop(im, [startCropX_orig*scaleX,startCropY_orig*scaleY,W-1,H-1]);
im= bsxfun(@times, im, cast(echoMask,class(im)));
