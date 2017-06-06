function writeCine2Folder(cine,addr,W,H, rgb, mask_crop, removeECGFlag)

meanImageW_orig = 500; meanImageH_orig = 370;
startCropX_orig = 68; startCropY_orig = 82;
scaleX = W/400;
scaleY = H/267;

echoMask = imread('masks/manualMask.bmp');
echoMask(echoMask>1) = 1;
echoMask = imresize(echoMask, size(echoMask).*[scaleY,scaleX]);

for f = 1:size(cine,4)
    im = cine(:,:,:,f);
    if rgb == false                
        im = rgb2gray(im);
    end
    
    if removeECGFlag
        im = removeECG(im);
    end
   
    im = imresize(im, [meanImageH_orig*scaleY,meanImageW_orig*scaleX]);                        
    if mask_crop    
        im = imcrop(im, [startCropX_orig*scaleX,startCropY_orig*scaleY,W-1,H-1]);    
        im= bsxfun(@times, im, cast(echoMask,class(im)));
    end

%     figure; imshow(im);
    imwrite(im, [addr '_' num2str(f,'%03.0f') '.png']);
end