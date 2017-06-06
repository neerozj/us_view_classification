clc
clear all
close all

addpath('functions')

src_fld = '/media/neeraj/pdf/cardiac_dys/DiastolicDysfunction_1731_2017.3.29/MatAnon/';
src_file = [src_fld 'file_list_complex_sortedFixed.csv'];
fid = fopen(src_file);
patients = textscan(fid,'%s','Delimiter','\n');

patients = patients{1,1};
fclose(fid);

dst_fld = [src_fld 'only_scored_cropped'];
mkdir(dst_fld);

for i = 1:size(patients,1)
    if mod(i,10) == 0
        disp(i);
    end
    strs = strsplit(patients{i}, ',');    
    file = strs{1};
    load(file);
    cine = Patient.DicomImage;
    
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
%         imshow(temp);
    end
    Patient.DicomImage = new_cine;
    
    dest_file = strrep(file, 'only_scored', 'only_scored_cropped');
    pathstr = fileparts(dest_file);
    mkdir(pathstr);
    save(dest_file, 'Patient');
    
end

