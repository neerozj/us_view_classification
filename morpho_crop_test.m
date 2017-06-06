function [patientID,studyDate,is_doppler, tmp_folder]  = morpho_crop_test(srcMat)
%srcMat = '/media/neeraj/pdf/cardiac_dys/DiastolicDysfunction_1731_2017.3.29/MatAnon/00473491/049_1.2.840.113619.2.98.8523.1287037052.0.1149.512.mat';
is_doppler=0;
load(srcMat);
cine = Patient.DicomImage;
patientID = Patient.DicomInfo.PatientID;
studyDate = Patient.DicomInfo.StudyDate;

%create a temporary folder
tmp_folder = fullfile(pwd,'tmp_folder');
if ~exist(tmp_folder)
    mkdir(fullfile(tmp_folder))
end


%single frame
if (size(cine,4) == 1)
    
    return
    
end
%doppler

imRGB = cine(:,:,:,1);
try
    [masked, mask] = maskEcho_convexhull(imRGB);
catch
    return
end
sumv = sum(mask,1);
sumh = sum(mask,2);
masked(sumh==0,:) = [];
masked(:,sumv==0) = [];
croppedFinal = cropMiddleSq(masked);

new_cine = uint8(zeros(size(croppedFinal,1), ...
    size(croppedFinal,2),...
    size(cine,4)));
for frame = 1:size(cine,4)
    try
        temp = maskEcho_convexhull(cine(:,:,:,frame));
    catch
        return
    end
    temp(sumh==0,:) = [];
    temp(:,sumv==0) = [];
    temp = cropMiddleSq(temp);
    new_cine(:,:,frame) = temp;
    
end
is_doppler = isDoppler(cine);
if is_doppler==1
    tag = 'Doppler_Image';
else
    tag= 'Normal_Image';
end

for f =1:size(new_cine,3)
    im = new_cine(:,:,f);
    filename = strcat(tag, sprintf('%03d',f),'.png');
    newImage = im;
    imwrite(newImage, fullfile(tmp_folder, filename))
    
end

end




