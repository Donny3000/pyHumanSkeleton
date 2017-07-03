function normalizedMat = normalizeImage(img)
sum=0;
count=0;
[m,n] = size(img);

for i=1:m
    for j=1:n
        l=img(i,j);
        if l~=0
        sum=sum+l;
        count=count+1;
        end
    end
end
mean = sum / count;

sum=0;
count=0;
for i=1:m
    for j=1:n
        l=img(i,j);
        if l~=0
            if l>mean
                l=0;
                img(i,j)=0;
            end
        sum=sum+l;
        count=count+1;
        end
    end
end


mean = sum / count;
var=0;
for i=1:m
    for j=1:n
        l=img(i,j);
        if l~=0
        f=l-mean;
        var=var+f^2;
        end
    end
end

sig = sqrt(var / count);
if sig<0.1/255
    sig=0.1/255;
end
numSDevs = 3.0;
meanGrey=(127.0 / 255.0) * 256.0;
halfrange = numSDevs * sig;
%meanGrey=0;
%halfrange=0;
range = numSDevs * sig *2* (1.0 / 256.0); 

for i=1:m
    for j=1:n
        l=img(i,j);
         
        if l==0
           img(i,j)=meanGrey;
        end
        if l~=0
            f=(l+halfrange)/range;
            if f>256
                f=256.0;
            end
            if f<0
                f=0;
            end

            img(i,j)= floor(f);
        end
    end
 end
normalizedMat=img;
end