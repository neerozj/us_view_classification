clc
clear all
close all
w = warning ('off','all');

srcFolder = '/media/neeraj/pdf/cardiac_dys/';
addpath(srcFolder);
srcMatFolder = '/media/neeraj/pdf/cardiac_dys/DiastolicDysfunction_1731_2017.3.29/MatAnon/';
destFolder = '/media/neeraj/pdf/cardiac_dys/view_classification/';

if ~exist(fullfile('/media/neeraj/pdf/cardiac_dys', 'view_classification'))
    mkdir([destFolder])
end
if ~exist(fullfile(destFolder,'AP4/' ))
    mkdir([destFolder 'AP4/'])
end
if ~exist(fullfile(destFolder,'AP2/' ))
    mkdir([destFolder 'AP2/'])
end
if ~exist(fullfile(destFolder,'none/' ))
    mkdir([destFolder 'none/'])
end
if ~exist(fullfile(destFolder,'doppler/' ))
    mkdir([destFolder 'doppler/'])
end

%% read test images from folder
TotalDataFolders = dir(srcMatFolder);
%%
echoMask = imread(['masks/' 'manualMask.bmp']);
echoMask(echoMask>1) = 1   ;

%%
cntCase = s150;
for i = cntCase+2:numel(TotalDataFolders)
    studyFolder = TotalDataFolders(i).name;
    cntStudyAP4 = 0;
    cntStudyAP2 = 0;
    cntStudyNone = 0;
    cntStudyDoppler=0;
    if (strcmpi(studyFolder,'.') || strcmpi(studyFolder,'..'))~=1
        matFiles = dir(fullfile(srcMatFolder,studyFolder, '*.mat'));
        for k  = 1:numel(matFiles)
            
            study_path = (fullfile(srcMatFolder, studyFolder,matFiles(k).name));
            [cine, isDoppler, patientId, studyDate] = morpho_crop(study_path);
            if isempty(cine)
                continue
            end
   
      
            
            %doppler
            if isDoppler==1
                cntStudyDoppler = cntStudyDoppler +1;
               
                caseStudyFolderName = strcat(patientId,'_', studyDate,'_', sprintf('%03d',cntStudyDoppler));
                if ~exist(fullfile(destFolder, 'doppler',caseStudyFolderName))
                    mkdir(fullfile(destFolder, 'doppler', caseStudyFolderName))
                end
                for f =1:size(cine,3)
                    im = cine(:,:,f);
                    filename = strcat('doppler_Image', sprintf('%03d',f),'.png');
                    newImage = im;
                    imwrite(newImage, fullfile(destFolder, 'doppler',caseStudyFolderName,filename))
                    
                    
                end
                
                continue;
            end
            
            
 
            
            for f = 1:5
                im = cine(:,:,f);
                imshow(im)
                pause(0.1)
            end
            
            
            
            loop = true;
            while(loop)
                prompt = strcat('Specify the view case_',...
                    num2str(cntCase),': AP4 as (1), Ap2 as (2), none as (3):');
                view = input(prompt);
                if view==1 || view==2 || view==3
                    loop = false;
                    
                else
                    fprintf('Try again, please enter correct value\n')
                    
                end
            end
            
            
            if view ==1
                cntStudyAP4 = cntStudyAP4 +1;
                caseStudyFolderName = strcat(patientId,'_', studyDate,'_', sprintf('%03d',cntStudyAP4));
                if ~exist(fullfile(destFolder, 'AP4',caseStudyFolderName))
                    mkdir(fullfile(destFolder, 'AP4', caseStudyFolderName))
                end
            end
            if view ==2
                cntStudyAP2 = cntStudyAP2 +1;
                caseStudyFolderName = strcat(patientId,'_', studyDate,'_', sprintf('%03d',cntStudyAP2));
                if ~exist(fullfile(destFolder, 'AP2',caseStudyFolderName))
                    mkdir(fullfile(destFolder, 'AP2', caseStudyFolderName))
                end
                
            end
            if view ==3
                cntStudyNone = cntStudyNone +1;
                caseStudyFolderName = strcat(patientId,'_', studyDate,'_', sprintf('%03d',cntStudyNone));
                if ~exist(fullfile(destFolder, 'none',caseStudyFolderName))
                    mkdir(fullfile(destFolder, 'none', caseStudyFolderName))
                end
                
            end
            
            for f = 1:size(cine,3)
                im = cine(:,:,f);
                %save the files to the folder
                newImage = im;
                
                if view ==1
                    filename = strcat('AP4_Image', sprintf('%03d',f),'.png');
                    imwrite(newImage, fullfile(destFolder, 'AP4',caseStudyFolderName,filename))
                    
                end
                
                if view ==2
                    filename = strcat('AP2_Image', sprintf('%03d',f),'.png');
                    imwrite(newImage, fullfile(destFolder, 'AP2',caseStudyFolderName,filename))
                end
                if view ==3'
                    filename = strcat('none_Image', sprintf('%03d',f),'.png');
                    imwrite(newImage, fullfile(destFolder, 'none',caseStudyFolderName,filename))
                end
                
                
                
            end
            
            
            disp(i)
        end
    end
    cntCase = cntCase+1;
end





