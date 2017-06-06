clear all
close all
srcFolder = '/media/neeraj/pdf/cardiac_dys/';
addpath(srcFolder);
srcMatFolder = '/media/neeraj/pdf/cardiac_dys/DiastolicDysfunction_1731_2017.3.29/MatAnon/';

cntCase = 1;
TotalDataFolders = dir(srcMatFolder);
for i = cntCase+2:numel(TotalDataFolders)
    studyFolder = TotalDataFolders(i).name;
    if (strcmpi(studyFolder,'.') || strcmpi(studyFolder,'..'))~=1
        matFiles = dir(fullfile(srcMatFolder,studyFolder, '*.mat'));
        for k  = 1:numel(matFiles)
            
            load(fullfile(srcMatFolder, studyFolder,matFiles(k).name));
            cine = Patient.DicomImage;
            
            %single frame
            if (size(cine,4) == 1)
                continue;
            end
            %doppler
            if isDoppler(cine)
                continue
            end
            
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
                imshow(temp);pause(0.2)
            end
        end
    end
end