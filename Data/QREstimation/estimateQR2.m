function [ Q, R ] = estimateQR2( joints, est )
[two, seven, s]=size(joints);
%Check input size
if two~=2
    error('input size wrong');
end

if seven~=7
    error('input size wrong');
end
%Setting model
F=[1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1];
%H=[1 0 0 0; 0 1 0 0];
%state vector
X=zeros(4,7,s);
X(1:2,:,:)=joints;

for i=2:s
    X(3,:,i)=joints(1,:,i)-joints(1,:,i-1);
    X(4,:,i)=joints(2,:,i)-joints(2,:,i-1);
end
%output
R=zeros(4,4,7);
Q=zeros(2,2,7);
for i=1:7
    for j=2:s
        R(:,:,i)=R(:,:,i)+(X(:,i,j)-F*X(:,i,j-1))*(X(:,i,j)-F*X(:,i,j-1))';
    end
    R(:,:,i)=R(:,:,i)/(s-1);
    for j=1:s
        Q(:,:,i)=Q(:,:,i)+(est(:,i,j)-joints(:,i,j))*(est(:,i,j)-joints(:,i,j))';
    end
    Q(:,:,i)=Q(:,:,i)/s;
end


end

