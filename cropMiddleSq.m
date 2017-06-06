function c = cropMiddleSq(im)
[rows,cols] = size(im);

cropSize = min([rows,cols]);
padR = max(round((rows-cropSize)/2),1);
padC = max(round((cols-cropSize)/2),1);

c = imcrop(im, [padC, padR, cropSize-1, cropSize-1]);

end