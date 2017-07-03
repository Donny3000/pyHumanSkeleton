clear all;
%close all;

addpath('C:\Users\iniLabs\Documents\matlab_workspace\C3D');
addpath('C:\Users\iniLabs\Documents\jAER\scripts\matlab');
addpath('C:\Users\iniLabs\Documents\jAER\scripts\matlab\AedatTools');

test=c3dserver;



%startDir = dir('C:\Users\iniLabs\Documents\HumanTracking\DataSetViconDVS\DVSwithVicon\23052017\Data\DVSmovies\Angelina\');
startDir = dir('G:\05062017\Data\MiXue\');
startDir=startDir(~ismember({startDir.name},{'.','..'}));
myDir = find(vertcat(startDir.isdir));
% for h=1:length(startDir)
% if startDir(h).isdir &&~strcmpi(startDir(h).name, '..')&& ~strcmpi(startDir(h).name, '.')
%     myDir(h)=startDir(h)
%myDir = find(contains(vertcat(startDir.name),'session'));
for l=1:length(myDir)
    a=strcat(startDir(myDir(l)).folder,'\');
    b=strcat(a,startDir(myDir(l)).name);
    currentDirFile=strcat(b,'\*.aedat');
    files = dir(currentDirFile);
    foldername=strcat('session',int2str(l));
    mkdir (foldername);
    currentFolder = pwd;
    filecounter=0;
    cd(currentFolder);
    cd (foldername);
    for f=1:length(files)
   % for file = files'
        filecounter=filecounter+1;
        folder = strcat(files(f).folder,'\');
        
        mkdir (files(f).name);
        cd (files(f).name);
        %c3d=strcat('session',l);
        c3d=strcat(int2str(l),int2str(0));
        c3d=strcat(c3d,int2str(filecounter));
        openc3d(test, 0, strcat('C:\Users\iniLabs\Documents\HumanTracking\DataSetViconDVS\DVSwithVicon\05062017\Data\C3Dfiles\MiXue\session',c3d));
        XYZPOS = get3dtargets(test, 0);
        
        aedat = ImportAedat(folder,files(f).name);
        starttime= min(aedat.data.special.timeStamp);
        stoptime= max(aedat.data.special.timeStamp);
        
        if(abs(starttime-stoptime)<10)    
           stoptime=  max(aedat.data.polarity.timeStamp)-1;
        end
        % numplot=(stoptime-starttime)*25/1000000;
        % PlotFrame(aedat, numplot, 'events', starttime, stoptime)
        eventPerFrame=10000;
        nbcam=3;
        
        str=erase(files(f).name, '.aedat');
        Events2Frames(aedat,eventPerFrame*nbcam,starttime, stoptime,str, XYZPOS);
        
        % construct the video
        imageNames = dir(fullfile('*.png'));
        imageNames = {imageNames.name}';
        outputVideo = VideoWriter(strcat(str,'.avi'),'Uncompressed AVI');
        open(outputVideo);
        for ii = 1:length(imageNames)
           m = imread(imageNames{ii});
           writeVideo(outputVideo,m)
        end
        close(outputVideo)
        
        cd ..
    end
    cd ..
end