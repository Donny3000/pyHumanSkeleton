function [] = Events2Frames(aedat, eventperframes, starttime, stoptime, fileName, XYZPOS)

addpath('C:\Users\iniLabs\Documents\matlab_workspace\C3D');

sx=346;
sy=260;

img=zeros(sy, sx*4);
if(isfield(XYZPOS, 'head'))
    head=XYZPOS.head;
else
    head=XYZPOS.Head;
end
if(isfield(XYZPOS,'shoulderR'))
    shouldR=XYZPOS.shoulderR;
    shouldL=XYZPOS.shoulderL;
else
    shouldR=XYZPOS.shouldR;
    shouldL=XYZPOS.shouldL;
end
elbowR=XYZPOS.elbowR;
elbowL=XYZPOS.elbowL;
if(isfield(XYZPOS,'hipR'))
    hipR=XYZPOS.hipR;
    hipL=XYZPOS.hipL;
else
    hipR=XYZPOS.HipR;
    hipL=XYZPOS.HipL;
end
handR=XYZPOS.handR;
handL=XYZPOS.handL;
kneeR=XYZPOS.kneeR;
kneeL=XYZPOS.kneeL;
footR=XYZPOS.footR;
footL=XYZPOS.footL;

pose=zeros(13,3);

% nbJoints=test.GetNumber3DPoints;
% fRate = test.GetVideoFrameRate;
% timestamp=0:1/fRate:size(XYZPOS.head);
statind=abs(aedat.data.polarity.timeStamp-starttime);
stopind=abs(aedat.data.polarity.timeStamp-stoptime);
startIndex=find(statind>1);
stopIndex=find(stopind>=1);

pol=aedat.data.polarity.polarity;
x=aedat.data.polarity.x;
y=sy-aedat.data.polarity.y+1;
cam=aedat.data.polarity.cam;
X=(sx-x)+cam*sx+1;
counter=0;
for i=startIndex(1):stopIndex(1)
    
    if (counter<eventperframes)
        coordx=X(i);
        coordy=y(i);
%         img(coordy,coordx)=1;
        if (coordx<4*sx && coordy<sy)&&((coordx>810||coordx<780)||(coordy>145||coordy<115))
            
           img(coordy,coordx)= img(coordy,coordx)+1;
           if((coordx>sx*3||coordx<sx*2))
               counter=counter+1;
           end
            %print('%%%%%');
            %tic;
           % if (x(i)~=800 && x(i)~=798 && x(i)~=799 ) && (y(i)~=128 && y(i)~=126 && y(i)~=125 && y(i)~=124)
%                 img(coordy,coordx)=img(coordy,coordx)+1
        end
    end
    
    if (counter==eventperframes)
        normedimg=normalizeImage(img);
        I = mat2gray(normedimg);
        imshow(I);
        j=i-startIndex(1);
        k=floor(j*1*0.0001);
        if k>length(head)
            break;
            %k=length(head);
        end
        name=strcat(int2str(j),'.png');
        imwrite(I,name)
        % save(name, 'I');
        % take the pose coordinates
       
        pose(1,:)=head(k,:);
        pose(2,:)=shouldR(k,:);
        pose(3,:)=shouldL(k,:);
        pose(4,:)=elbowR(k,:);
        pose(5,:)=elbowL(k,:);
        pose(6,:)=hipR(k,:);
        pose(7,:)=hipL(k,:);
        pose(8,:)=handR(k,:);
        pose(9,:)=handL(k,:);
        pose(10,:)=kneeR(k,:);
        pose(11,:)=kneeL(k,:);
        pose(12,:)=footR(k,:);
        pose(13,:)=footL(k,:);
        
        posename=int2str(j);
        save(posename, 'pose');
        matlab2xmlOfVideoFrame(pose, strcat(fileName,'.xml'), posename, 'a');
        counter=0;
        img=zeros(sy,sx*4);
        
    end
    if (counter>eventperframes)
        disp('error');
    end
    
end
s = pwd;
f=strcat(fileName,'.xml');
s= strcat(s,'/' );
file = fopen(strcat(s,f), 'a');
        fprintf(file, '</root>');

