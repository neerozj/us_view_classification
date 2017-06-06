clc
clear all
close all

addpath('/home/amir/caffe/matlab');
caffe.set_mode_gpu();
caffe.reset_all()
srcFolder = '/home/amir/echoProject/AP2_Ap4_ViewClassifier/';
addpath(srcFolder);
srcMatFolder = '/media/truecrypt1/AnonPat_NotNormalized/';
destFolder = '/media/truecrypt1/classified_views_4Classes/';

mkdir([destFolder])
mkdir([destFolder 'AP2/'])
mkdir([destFolder 'AP4/'])
mkdir([destFolder 'none/'])
mkdir([destFolder 'doppler/'])
%%
net = caffe.Net([srcFolder 'network_deploy.prototxt'], 'test');
iteration = 100000;
weightsModelFile = [srcFolder 'ap2_ap4_classifier.caffemodel'];
net.copy_from(weightsModelFile);
%%
mean_im  = caffe.io.read_mean([srcFolder '/mean_train_image.binaryproto']);
size(mean_im)
mean_im = cropMiddleSq(mean_im);
%%
accuracy = [];
numClasses = 2;
H = 267;
W = 267;

%% read test images from folder
files = dir([srcMatFolder '*.mat']);
%%
echoMask = imread(['masks/' 'manualMask.bmp']);
echoMask(echoMask>1) = 1    ;
% 
%%
for i = 1:numel(files)
    load([srcMatFolder files(i).name]);
    ap2Flag = -1;
    ap4Flag = -1;
    
    for j = 1:numel(patient.study)         
        if isfield(patient.study(j), 'cleanVolume')
            cine = patient.study(j).cleanVolume;
        elseif isfield(patient.study(j), 'volume')
            cine = patient.study(j).volume;
        else
            continue; %doesn't contain any volume info
        end
        
        %single frame
        if (size(cine,4) == 1) 
            continue;
        end
        
        %doppler
        if isDoppler(cine)
            fld = [destFolder 'doppler/' files(i).name(1:end-4) '_' num2str(j)];
            mkdir(fld)            
            sprintf('%d %d %s', i, j, 'doppler')
            writeCine2Folder(cine,[fld '/' files(i).name(1:end-4)],600,400,true, false, false);
            continue;
        end
        
        
        framesView = zeros(size(cine,4),2);
        
        for f = 1:size(cine,4)
            im = cine(:,:,:,f);
            im = mask_and_crop_echo(im, 400, 267);
            
            im = im';
            
            im = cropMiddleSq(im);  %267x267
            
            imNetwork = single(im)-mean_im;
            
            res = net.forward({imNetwork});
            res = res{1};     
            if max(res(1),res(2)) < 2 %abs(res(1) - res(2)) < 1.5
                framesView(f,1) = 0 ;
            else
                if res(1) > res(2)
                    framesView(f,1) = 1;
                else
                    framesView(f,1) = 2;
                end    
            end
            framesView(f,2) = max(res(1),res(2));
        end
        view = mode(framesView(:,1));
        confidence = mean(framesView(:,2));
        if (view == 1) && (confidence > ap2Flag)
            fld = [destFolder 'AP2/' files(i).name(1:end-4) '_' num2str(j)]; 
            for t = 1:j-1
                if exist([destFolder 'AP2/' files(i).name(1:end-4) '_' num2str(t)],'dir')==7
                    movefile([destFolder 'AP2/' files(i).name(1:end-4) '_' num2str(t)],...
                        [destFolder 'none/' files(i).name(1:end-4) '_' num2str(t)]);
                end
            end
            mkdir(fld)            
            sprintf('%d %d %s', i, j, 'AP2')
            ap2Flag = confidence;
        elseif (view == 2) && (confidence > ap4Flag)      
            fld = [destFolder 'AP4/' files(i).name(1:end-4) '_' num2str(j)];
            for t = 1:j-1
                if exist([destFolder 'AP4/' files(i).name(1:end-4) '_' num2str(t)],'dir')==7
                    movefile([destFolder 'AP4/' files(i).name(1:end-4) '_' num2str(t)],...
                        [destFolder 'none/' files(i).name(1:end-4) '_' num2str(t)]);
                end
            end            
            mkdir(fld)
            sprintf('%d %d %s', i, j, 'AP4')
            ap4Flag = confidence;            
        else
            fld = [destFolder 'none/' files(i).name(1:end-4) '_' num2str(j)];
            mkdir(fld)
            sprintf('%d %d %s', i, j, 'none')
        end
        writeCine2Folder(cine,[fld '/' files(i).name(1:end-4)],640,480,true, false, false);
    end
    
    if mod(i,100) == 0
        sprintf('%d/%d', i, numel(files))
    end
    disp(i)
end

