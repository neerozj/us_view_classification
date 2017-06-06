function d = isDoppler(cine)

temp = cine(:,:,:,1);
maskRed = (temp(:,:,1)>10*temp(:,:,2)) & (temp(:,:,1)>10*temp(:,:,3)) &...
     temp(:,:,1) ~= 0 & temp(:,:,2) ~= 0 & temp(:,:,3) ~= 0;
if sum(sum(maskRed)) ~= 0         
    d = true;
else
    d = false;
end