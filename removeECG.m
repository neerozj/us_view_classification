function out = removeECG(in)
if size(in, 3) < 3
    out = in;
    return
end
    

%remove ECG
mask = in(:,:,2)>1.45*in(:,:,1) | in(:,:,2)>1.45*in(:,:,3);
in= bsxfun(@times, in, cast(~mask,class(in)));

%remove marks
marksMask = (in > 235);
marksMask = marksMask(:,:,1)&...
    marksMask(:,:,2)&...
    marksMask(:,:,3);
out = rgb2gray(in);
med = medfilt2(out, [3,3]);

out(marksMask) = med(marksMask);

end