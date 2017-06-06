function [process_cine, is_doppler, patientID, studyDate] = morpho_crop(path)
%srcMatFolder = '/media/neeraj/pdf/cardiac_dys/DiastolicDysfunction_1731_2017.3.29/MatAnon/';
srcMat = path;
process_cine = [] ;
is_doppler =0;

load(srcMat);
cine = Patient.DicomImage;
patientID = Patient.DicomInfo.PatientID;
studyDate = Patient.DicomInfo.StudyDate;


%single frame
if (size(cine,4) == 1)
    
    return
    
end
%doppler


imRGB = cine(:,:,:,1);
[masked, mask] = maskEcho_convexhull(imRGB);
sumv = sum(mask,1);
sumh = sum(mask,2);
masked(sumh==0,:) = [];
masked(:,sumv==0) = [];
croppedFinal = cropMiddleSq(masked);

new_cine = uint8(zeros(size(croppedFinal,1), ...
    size(croppedFinal,2),...
    size(cine,4)));
for frame = 1:size(cine,4)
    temp = maskEcho_convexhull(cine(:,:,:,frame));
    temp(sumh==0,:) = [];
    temp(:,sumv==0) = [];
    temp = cropMiddleSq(temp);
    new_cine(:,:,frame) = temp;
    
end
process_cine = new_cine;
if isDoppler(cine)
    is_doppler=1;
    
end
end

