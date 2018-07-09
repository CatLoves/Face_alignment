function varargout = shapeGt( action, varargin )
%
% Wrapper with utils for handling shape as list of landmarks
%
% shapeGt contains a number of utility functions, accessed using:
%  outputs = shapeGt( 'action', inputs );
%
% USAGE
%  varargout = shapeGt( action, varargin );
%
% INPUTS
%  action     - string specifying action
%  varargin   - depends on action
%
% OUTPUTS
%  varargout  - depends on action
%
% FUNCTION LIST
% 
%%%% Model creation and visualization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   shapeGt>createModel, shapeGt>draw
%
%%%% Shape composition, inverse, distances, projection %%%%%%%%%%%%%%%
% 
%   shapeGt>compose,shapeGt>inverse, shapeGt>dif, shapeGt>dist
%   shapeGt>compPhiStar, shapeGt>reprojectPose, shapeGt>projectPose_wrt_bbox
% 
%%%% Shape-indexed features computation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   shapeGt>ftrsGenIm,shapeGt>ftrsCompIm
%   shapeGt>ftrsGenDup,shapeGt>ftrsComDup
%   shapeGt>ftrsOcclMasks, shapeGt>codifyPos
%   shapeGt>getLinePoint, shapeGt>getSzIm
%   
%%%%% Random shape generation for initialization  %%%%%%%%%%%%%%%%%%%%
%   
%   shapeGt>initTr, shapeGt>initTest
%
% EXAMPLES
%
%%create COFW model 
%   model = shapeGt( 'createModel', 'cofw' );
%%draw shape on top of image
%   shapeGt( 'draw',model,Image,shape);
%%compute distance between two set of shapes (phis1 and phis2)
%   d = shapeGt( 'dist',model,phis1,phis2);
varargout = cell(1,max(1,nargout));
[varargout{:}] = feval(action,varargin{:});
end

function model = createModel( type )
global NUM_LANDMARKS;
% Create shape model (model is necessary for all other actions).
    model=struct('num_landmarks',0,'dimension',0,'isFace',1,'name',[]);
    switch type
        case 'cofw' % COFW dataset (29 landmarks: X,Y,V)
            model.num_landmarks=29;model.dimension=model.num_landmarks*3; model.name='cofw';
        case 'lfpw' % LFPW dataset (29 landmarks: X,Y)
            model.num_landmarks=29;model.dimension=model.num_landmarks*2; model.name='lfpw';
        case 'helen' % HELEN dataset (194 landmarks: X,Y)
            model.num_landmarks=194;model.dimension=model.num_landmarks*2;model.name='helen';
        case 'lfw' % LFW dataset (10 landmarks: X,Y)
            model.num_landmarks=10;model.dimension=model.num_landmarks*2; model.name='lfw';
        case 'pie' %Multi-pie & 300-Faces in the wild dataset (68 landmarks)
            model.num_landmarks=68;model.dimension=model.num_landmarks*2;model.name='pie';
        case 'apf' %anonimous portrait faces
            model.num_landmarks=55;model.dimension=model.num_landmarks*2;model.name='apf';
        otherwise
            error('unknown type: %s',type);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = draw( model, Is, phis, varargin )
% Draw shape with parameters phis using model on top of image Is.
dfs={'n',25, 'clrs','gcbm', 'drawIs',1, 'lw',10, 'is',[]};
[n,cs,drawIs,lw,is]=getPrmDflt(varargin,dfs,1);

% display I
if(drawIs), im(Is); colorbar off; axis off; title(''); axis('ij'); end%clf
% special display for face model (draw face points)
hold on,
if( isfield(model,'isFace') && model.isFace ),
    [N,dimension]=size(phis);
    if(strcmp(model.name,'cofw')),
        %WITH OCCLUSION
        num_landmarks = dimension/3;
        for n=1:N
            occl=phis(n,(num_landmarks*2)+1:num_landmarks*3);
            vis=find(occl==0);novis=find(occl==1);
            plot(phis(n,vis),phis(n,vis+num_landmarks),'g.',...
                'MarkerSize',lw);
            h=plot(phis(n,novis),phis(n,novis+num_landmarks),'r.',...
                'MarkerSize',lw);
        end
    else
        %REGULAR
        if(N==1),cs='g';end, num_landmarks = dimension/2;
        for n=1:N
            h=plot(phis(n,1:num_landmarks),phis(n,num_landmarks+1:num_landmarks*2),[cs(n) '.'],...
                'MarkerSize',lw);
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pos=ftrsOcclMasks(xs)
%Generate 9 occlusion masks for varied feature locations
pos=cell(9,1);
for m=1:9
    switch(m)
        case 1,pos{m}=(1:numel(xs(:,1)))';
        case 2,%top half
            pos{m}=find(xs(:,2)<=0);
        case 3,%bottom half
            pos{m}=find(xs(:,2)>0);
        case 4,%right
            pos{m}=find(xs(:,1)>=0);
        case 5,%left
            pos{m}=find(xs(:,1)<0);
        case 6,%right top diagonal
            pos{m}=find(xs(:,1)>=xs(:,2));
        case 7,%left bottom diagonal
            pos{m}=find(xs(:,1)<xs(:,2));
        case 8,%left top diagonal
            pos{m}=find(xs(:,1)*-1>=xs(:,2));
        case 9,%right bottom diagonal
            pos{m}=find(xs(:,1)*-1<xs(:,2));
    end
end
end

function ftrData = ftrsGenDup( model, varargin )
% Generate random shape indexed features, relative to
% two landmarks (points in a line, RCPR contribution)
% Features are then computed using frtsCompDup
%
% USAGE
%  ftrData = ftrsGenDup( model, varargin )
%
% INPUTS
%  model    - shape model (see createModel())
%  varargin - additional params (struct or name/value pairs)
%   .type     - [2] feature type (1 or 2)
%   .F        - [100] number of ftrs to generate
%   .radius   - [2] sample initial x from circle of given radius
%   .nChn     - [1] number of image channels (e.g. 3 for color images)
%   .pids     - [] part ids for each x
%
% OUTPUTS
%  ftrData  - struct containing generated ftrs
%   .type     - feature type (1 or 2)
%   .F        - total number of features
%   .nChn     - number of image channels
%   .xs       - feature locations relative to unit circle
%   .pids     - part ids for each x
%
% EXAMPLE
%
% See also shapeGt>ftrsCompDup

dfs={'type',4,'F',20,'radius',1,'nChn',3,'pids',[],'mask',[]};
[type,F,radius,nChn,pids,mask]=getPrmDflt(varargin,dfs,1);
F2=max(100,ceil(F*1.5));
xs=[];num_landmarks=model.num_landmarks;
while(size(xs,1)<F),
    %select two random landmarks
    xs(:,1:2)=randint2(F2,2,[1 num_landmarks]);
    %make sure they are not the same
    neq = (xs(:,1)~=xs(:,2));
    xs=xs(neq,:);
end
xs=xs(1:F,:);
%select position in line
xs(:,3)=(2*radius*rand(F,1))-radius;
if(nChn>1),
    if(type==4),%make sure subbtractions occur inside same channel
        chns = randint2(F/2,1,[1 nChn]);
        xs(1:2:end,4) = chns; xs(2:2:end,4) = chns;
    else xs(:,4)=randint2(F,1,[1 nChn]);
    end
end
if(isempty(pids)), pids=floor(linspace(0,F,2)); end
ftrData=struct('type',type,'F',F,'nChn',nChn,'xs',xs,'pids',pids);
end

function ftrData = ftrsGenIm( model, pStart, varargin )
% Generate random shape indexed features,
% relative to closest landmark (similar to Cao et al., CVPR12)
% Features are then computed using frtsCompIm
%
% USAGE
%  ftrData = ftrsGenIm( model, pStart, varargin )
%
% INPUTS
%  model    - shape model (see createModel())
%  pStart    - average shape (see initTr)
%  varargin - additional params (struct or name/value pairs)
%   .type     - [2] feature type (1 or 2)
%   .F        - [100] number of ftrs to generate
%   .radius   - [2] sample initial x from circle of given radius
%   .nChn     - [1] number of image channels (e.g. 3 for color images)
%   .pids     - [] part ids for each x
%
% OUTPUTS
%  ftrData  - struct containing generated ftrs
%   .type     - feature type (1 or 2)
%   .F        - total number of features
%   .nChn     - number of image channels
%   .xs       - feature locations relative to unit circle
%   .pids     - part ids for each x
%
% EXAMPLE
%
% See also shapeGt>ftrsCompIm

dfs={'type',2,'F',20,'radius',1,'nChn',3,'pids',[],'mask',[]};
[type,F,radius,nChn,pids,mask]=getPrmDflt(varargin,dfs,1);
%Generate random features on image
xs1=[];
while(size(xs1,1)<F),
    xs1=rand(F*1.5,2)*2-1;
    xs1=xs1(sum(xs1.^2,2)<=1,:);
end
xs1=xs1(1:F,:)*radius;

if(strcmp(model.name,'cofw'))
    num_landmarks=size(pStart,2)/3;
else
    num_landmarks=size(pStart,2)/2;
end
%Reproject each into closest pStart landmark
xs=zeros(F,3);%X,Y,landmark
for f=1:F
    posX=xs1(f,1)-pStart(1:num_landmarks);
    posY=xs1(f,2)-pStart(num_landmarks+1:num_landmarks*2);
    dist = (posX.^2)+(posY.^2);
    [~,l]=min(dist);xs(f,:)=[posX(l) posY(l) l];
end
if(nChn>1),
    if(mod(type,2)==0),%make sure subbtractions occur inside same channel
        chns = randint2(F,1,[1 nChn]);
        xs(1:2:end,4) = chns; xs(2:2:end,4) = chns;
    else xs(:,4)=randint2(F,1,[1 nChn]);
    end
end
if(isempty(pids)), pids=floor(linspace(0,F,2)); end
ftrData=struct('type',type,'F',F,'nChn',nChn,'xs',xs,'pids',pids);
end

function [ftrs,occlD] = ftrsCompDup( model, phis, Is, ftrData,...
    imgIds, pStart, bboxes, occlPrm)
% Compute features from ftrsGenDup on Is 
%
% USAGE
%  [ftrs,Vs] = ftrsCompDup( model, phis, Is, ftrData, imgIds, pStart, ...
%       bboxes, occlPrm )
%
% INPUTS
%  model    - shape model
%  phis     - [MxR] relative shape for each image 
%  Is       - cell [N] input images [w x h x nChn] variable w, h
%  ftrData  - define ftrs to actually compute, output of ftrsGen
%  imgIds   - [Mx1] image id for each phi 
%  pStart   -  [1xR] average shape (see initTr) 
%  bboxes   - [Nx4] face bounding boxes 
%  occlPrm   - struct containing occlusion reasoning parameters
%    .nrows - [3] number of rows in face region
%    .ncols - [3] number of columns in face region
%    .nzones - [1] number of zones from where each regs draws
%    .Stot -  total number of regressors at each stage
%    .th - [0.5] occlusion threshold used during cascade
%
% OUTPUTS
%  ftrs     - [MxF] computed features
%  occlD    - struct containing occlusion info (if using full RCPR) 
%       .group    - [MxF] to which face region each features belong     
%       .featOccl - [MxF] amount of total occlusion in that area
%     
% EXAMPLE
%
% See also demoRCPR, shapeGt>ftrsGenDup
N = length(Is); num_landmarks=model.num_landmarks;
if(nargin<5 || isempty(imgIds)), imgIds=1:N; end
if(nargin<6 || isempty(pStart)),
    pStart=compPhiStar(model,phis,Is,0,[],[]);
end
M=size(phis,1); assert(length(imgIds)==M);nChn=ftrData.nChn;

if(size(bboxes,1)==length(Is)), bboxes=bboxes(imgIds,:); end

if(ftrData.type==3),
    FTot=ftrData.F;
    ftrs = zeros(M,FTot);
else
    FTot=ftrData.F;ftrs = zeros(M,FTot);
end
posrs = phis(:,num_landmarks+1:num_landmarks*2);poscs = phis(:,1:num_landmarks);
useOccl=occlPrm.Stot>1;
if(useOccl && (strcmp(model.name,'cofw')))
    occl = phis(:,(num_landmarks*2)+1:num_landmarks*3);
    occlD=struct('featOccl',zeros(M,FTot),'group',zeros(M,FTot));
else occl = zeros(M,num_landmarks);occlD=[];
end
%GET ALL POINTS
if(nargout>1)
    [csStar,rsStar]=getLinePoint(ftrData.xs,pStart(1:num_landmarks),...
        pStart(num_landmarks+1:num_landmarks*2));
    pos=ftrsOcclMasks([csStar' rsStar']);
end
%relative to two points
[cs1,rs1]=getLinePoint(ftrData.xs,poscs,posrs); 
nGroups=occlPrm.nrows*occlPrm.ncols;
%ticId =ticStatus('Computing feats',1,1,1);
for n=1:M
    img = Is{imgIds(n)}; [h,w,ch]=size(img);
    if(ch==1 && nChn==3), img = cat(3,img,img,img);
    elseif(ch==3 && nChn==1), img = rgb2gray(img);
    end
    cs1(n,:)=max(1,min(w,cs1(n,:)));
    rs1(n,:)=max(1,min(h,rs1(n,:)));
    
    % where are the features relative to bbox?
    if(useOccl && (strcmp(model.name,'cofw')))
        %to which group (zone) does each feature belong?
        occlD.group(n,:)=codifyPos((cs1(n,:)-bboxes(n,1))./bboxes(n,3),...
            (rs1(n,:)-bboxes(n,2))./bboxes(n,4),...
            occlPrm.nrows,occlPrm.ncols);
        %to which group (zone) does each landmark belong?
        groupF=codifyPos((poscs(n,:)-bboxes(n,1))./bboxes(n,3),...
            (posrs(n,:)-bboxes(n,2))./bboxes(n,4),...
            occlPrm.nrows,occlPrm.ncols);
        %NEW
        %therefore, what is the occlusion in each group (zone)
        occlAm=zeros(1,nGroups);
        for g=1:nGroups
            occlAm(g)=sum(occl(n,groupF==g));
        end
        %feature occlusion = sum of occlusion on that area
        occlD.featOccl(n,:)=occlAm(occlD.group(n,:));
    end
    
    inds1 = (rs1(n,:)) + (cs1(n,:)-1)*h;
    if(nChn>1), inds1 = inds1+(chs'-1)*w*h; end
    
    if(isa(img,'uint8')), ftrs1=double(img(inds1)')/255;
    else ftrs1=double(img(inds1)'); end
    
    if(ftrData.type==3),ftrs1=ftrs1*2-1; ftrs(n,:)=reshape(ftrs1,1,FTot);
    else ftrs(n,:)=ftrs1;
    end
    %tocStatus(ticId,n/M);
end
end

function group=codifyPos(x,y,nrows,ncols)
%codify position of features into regions
nr=1/nrows;nc=1/ncols;
%Readjust positions so that they falls in [0,1]
x=min(1,max(0,x));y=min(1,max(0,y)); 
y2=y;x2=x;
for c=1:ncols,
    if(c==1), x2(x<=nc)=1; 
    elseif(c==ncols), x2(x>=nc*(c-1))=ncols;
    else x2(x>nc*(c-1) & x<=nc*c)=c;
    end
end
for r=1:nrows,
    if(r==1), y2(y<=nr)=1; 
    elseif(r==nrows), y2(y>=nc*(r-1))=nrows;
    else y2(y>nr*(r-1) & y<=nr*r)=r;
    end 
end
group=sub2ind2([nrows ncols],[y2' x2']);
end

function [cs1,rs1]=getLinePoint(FDxs,poscs,posrs)
%get pixel positions given coordinates as points in a line between
%landmarks
%INPUT NxF, OUTPUT NxF
if(size(poscs,1)==1)%pStart normalized
    l1= FDxs(:,1);l2= FDxs(:,2);xs=FDxs(:,3);
    x1 = poscs(:,l1);y1 = posrs(:,l1);
    x2 = poscs(:,l2);y2 = posrs(:,l2);
    
    a=(y2-y1)./(x2-x1); b=y1-(a.*x1);
    distX=(x2-x1)/2; center_x_bbox= x1+distX;
    cs1=center_x_bbox+(repmat(xs',size(distX,1),1).*distX);
    rs1=(a.*cs1)+b;
else
    if(size(FDxs,2)<4)%POINTS IN A LINE (ftrsGenDup)
        %2 points in a line with respect to center_bbox
        l1= FDxs(:,1);l2= FDxs(:,2);xs=FDxs(:,3);
        %center_bbox
        muX = mean(poscs,2);
        muY = mean(posrs,2);
        poscs=poscs-repmat(muX,1,size(poscs,2));
        posrs=posrs-repmat(muY,1,size(poscs,2));
        %landmark
        x1 = poscs(:,l1);y1 = posrs(:,l1);
        x2 = poscs(:,l2);y2 = posrs(:,l2);
        
        a=(y2-y1)./(x2-x1); b=y1-(a.*x1);
        distX=(x2-x1)/2; center_x_bbox= x1+distX;
        cs1=center_x_bbox+(repmat(xs',size(distX,1),1).*distX);
        rs1=(a.*cs1)+b;
        cs1=round(cs1+repmat(muX,1,size(FDxs,1)));
        rs1=round(rs1+repmat(muY,1,size(FDxs,1)));
    end
end
end

function [ftrs,occlD] = ftrsCompIm( model, phis, Is, ftrData,...
    imgIds, pStart, bboxes, occlPrm )
% Compute features from ftrsGenIm on Is 
%
% USAGE
%  [ftrs,Vs] = ftrsCompIm( model, phis, Is, ftrData, [imgIds] )
%
% INPUTS
%  model    - shape model
%  phis     - [MxR] relative shape for each image 
%  Is       - cell [N] input images [w x h x nChn] variable w, h
%  ftrData  - define ftrs to actually compute, output of ftrsGen
%  imgIds   - [Mx1] image id for each phi 
%  pStart   -  [1xR] average shape (see initTr) 
%  bboxes   - [Nx4] face bounding boxes 
%  occlPrm   - struct containing occlusion reasoning parameters
%    .nrows - [3] number of rows in face region
%    .ncols - [3] number of columns in face region
%    .nzones - [1] number of zones from where each regs draws
%    .Stot -  total number of regressors at each stage
%    .th - [0.5] occlusion threshold used during cascade
%
% OUTPUTS
%  ftrs     - [MxF] computed features
%  occlD    - [] empty structure
%
% EXAMPLE
%
% See also demoCPR, shapeGt>ftrsGenIm, shapeGt>ftrsCompDup

N = length(Is); nChn=ftrData.nChn;
if(nargin<5 || isempty(imgIds)), imgIds=1:N; end
M=size(phis,1); assert(length(imgIds)==M);

[pStart,phis_norm_wrt_bbox,distPup,sz,bboxes] = compPhiStar( model, phis, Is, 10, imgIds, bboxes );

if(size(bboxes,1)==length(Is)), bboxes=bboxes(imgIds,:); end

F=size(ftrData.xs,1);ftrs = zeros(M,F);
useOccl=occlPrm.Stot>1;
if(strcmp(model.name,'cofw'))
    num_landmarks=size(phis,2)/3;occlD=[];
else
    num_landmarks=size(phis,2)/2;occlD=[];
end

%X,Y,landmark,Channel
rs = ftrData.xs(:,2);cs = ftrData.xs(:,1);xss = [cs';rs'];
ls = ftrData.xs(:,3);if(nChn>1),chs = ftrData.xs(:,4);end
%Actual phis positions
poscs=phis(:,1:num_landmarks);posrs=phis(:,num_landmarks+1:num_landmarks*2);
%get positions of key landmarks
posrs=posrs(:,ls);poscs=poscs(:,ls);
%Reference points
X=[pStart(1:num_landmarks);pStart(num_landmarks+1:num_landmarks*2)];
for n=1:M
    img = Is{imgIds(n)}; [h,w,ch]=size(img);
    if(ch==1 && nChn==3), img = cat(3,img,img,img);
    elseif(ch==3 && nChn==1), img = rgb2gray(img);
    end
    
    %Compute relation between phis_norm_wrt_bbox and pStart (scale, rotation)
    Y=[phis_norm_wrt_bbox(n,1:num_landmarks);phis_norm_wrt_bbox(n,num_landmarks+1:num_landmarks*2)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~,~,Sc,Rot] = translate_scale_rotate(Y,X);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Compute feature locations by reprojecting
    aux=Sc*Rot*xss;
    %Resize accordingly to bbox size
    szX=bboxes(n,3)/2;szY=bboxes(n,4)/2;
    aux = [(aux(1,:)*szX);(aux(2,:)*szY)];
    
    %Add to respective landmark
    rs1 = round(posrs(n,:)+aux(2,:));
    cs1 = round(poscs(n,:)+aux(1,:));
    
    cs1 = max(1,min(w,cs1)); rs1=max(1,min(h,rs1));
    
    inds1 = (rs1) + (cs1-1)*h;
    if(nChn>1), chs = repmat(chs,1,m); inds1 = inds1+(chs-1)*w*h; end
    
    if(isa(img,'uint8')), ftrs1=double(img(inds1)')/255;
    else ftrs1=double(img(inds1)'); end
    
    if(ftrData.type==1),
        ftrs1=ftrs1*2-1; ftrs(n,:)=reshape(ftrs1,1,F);
    else ftrs(n,:)=ftrs1;
    end
end
end

function [h,w]=getSzIm(Is)
%get image sizes
N=length(Is); w=zeros(1,N);h=zeros(1,N);
for i=1:N, [w(i),h(i),~]=size(Is{i}); end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function phis = compose( model, phis0, phis1, bboxes )
% Compose two shapes phis0 and phis1: phis = phis0 + phis1.
phis1 = projectPose_wrt_bbox(model,phis1,bboxes);
phis=phis0+phis1;
end

% Compute inverse of two shapes phis0 so that phis0+phis=phis+phis0=identity.
function phis = inverse_RCPR( model, phis0, bboxes )
    phis = -projectPose_wrt_bbox(model,phis0,bboxes);
end

function phis = inverse( model, phis0, bboxes )
    phis = -projectPose_wrt_bbox(model,phis0,bboxes);
end

% Input: model: model
%        phis_norm: Num_obs * dim(phi)
%        bboxes:    Num_obs * 4(left,upper,width,height)
% Output: Original image coordinates( Just raw image coordinates )
function phis1 = reprojectPose( model , phis_norm , bboxes)
[N,dimension]=size(phis_norm);
if(strcmp(model.name,'cofw'))
    num_landmarks = dimension/2;
else
    num_landmarks = dimension/2;
end

szX=bboxes(:,3)/2;      % 1/2 * width_bbox
szY=bboxes(:,4)/2;      % 1/2 * height_bbox
center_x_bbox = bboxes(:,1)+szX;
center_y_bbox = bboxes(:,2)+szY;
szX=repmat(szX,1,num_landmarks);
szY=repmat(szY,1,num_landmarks);
center_x_bbox=repmat(center_x_bbox,1,num_landmarks);
center_y_bbox=repmat(center_y_bbox,1,num_landmarks);
% if(strcmp(model.name,'cofw'))
%     phis1 = [ (phis_norm(:,1:num_landmarks).*szX)+center_x_bbox ...
%               (phis_norm(:,num_landmarks+1:num_landmarks*2).*szY)+center_y_bbox...
%                phis_norm(:,(num_landmarks*2)+1:num_landmarks*3)];
% else
    phis1 = [(phis_norm(:,1:num_landmarks).*szX)+center_x_bbox  ...  % from norm_true to true_img_coords
             (phis_norm(:,num_landmarks+1:num_landmarks*2).*szY)+center_y_bbox];
% end
    
end

% Input: (1)model (2)image coords of shape (3)bbox
% Output: phi_wrt_bbox
% Note: for just one image !!!
function phis = projectPose_wrt_bbox(model , phis , bboxes)
%project shape onto bounding box of object location
[N,dimension]=size(phis);
if( strcmp(model.name,'cofw') ) 
    num_landmarks=dimension/2;
else
    num_landmarks=dimension/2;
end
szX=bboxes(:,3)/2;
szY=bboxes(:,4)/2;
center_x_bbox=bboxes(:,1)+szX;
center_y_bbox=bboxes(:,2)+szY;
szX=repmat(szX,1,num_landmarks);
szY=repmat(szY,1,num_landmarks);
center_x_bbox=repmat(center_x_bbox,1,num_landmarks);
center_y_bbox=repmat(center_y_bbox,1,num_landmarks);
phis = [(phis(:,1:num_landmarks)-center_x_bbox)./szX (phis(:,num_landmarks+1:num_landmarks*2)-center_y_bbox)./szY];

end

function del = dif( phis0, phis1 )
% Compute diffs between phis0(i,:,t) and phis1(i,:) for each i and t.
[N,R,T]=size(phis0); 
assert(size(phis1,3)==1);
del = phis0-phis1(:,:,ones(1,1,T));
end

% distance between each point in phis0 and phis1 
% Relative to the distance between pupils in the image (phis1 = gt)
function [ds,dsAll] = dist( model, phis0, phis1 )
global NUM_LANDMARKS;
[N,R,T]=size(phis0); 
% dif: just phis0 - phis1( phis1 = gt )
del = dif(phis0,phis1);
num_landmarks = NUM_LANDMARKS;
assert( NUM_LANDMARKS > 0 & isscalar(NUM_LANDMARKS) );

%Distance between pupils
if(strcmp(model.name,'lfpw') || strcmp(model.name,'cofw'))
    distPup=sqrt(((phis1(:,17)-phis1(:,18)).^2) + ...
                 ((phis1(:,17+num_landmarks)-phis1(:,18+num_landmarks)).^2));
    distPup = repmat(distPup,[1,num_landmarks,T]);
elseif(strcmp(model.name,'lfw'))
    leyeX=mean(phis1(:,1:2),2);leyeY=mean(phis1(:,(1:2)+num_landmarks),2);
    reyeX=mean(phis1(:,7:8),2);reyeY=mean(phis1(:,(7:8)+num_landmarks),2);
    distPup=sqrt(((leyeX-reyeX).^2) + ((leyeY-reyeY).^2));
    distPup = repmat(distPup,[1,num_landmarks,T]);
elseif(strcmp(model.name,'helen'))
    leye = [mean(phis1(:,135:154),2) mean(phis1(:,num_landmarks+(135:154)),2)];
    reye = [mean(phis1(:,115:134),2) mean(phis1(:,num_landmarks+(115:134)),2)];
    distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
        ((reye(:,2)-leye(:,2)).^2));
    distPup = repmat(distPup,[1,num_landmarks,T]);
elseif(strcmp(model.name,'pie')) % See https://ibug.doc.ic.ac.uk/resources/300-W/
    leye = [mean(phis1(:,37),2) mean(phis1(:,num_landmarks+(37)),2)];
    reye = [mean(phis1(:,46),2) mean(phis1(:,num_landmarks+(46)),2)];
    distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
        ((reye(:,2)-leye(:,2)).^2));
    distPup = repmat(distPup,[1,num_landmarks,T]);
elseif(strcmp(model.name,'apf'))
    leye = [mean(phis1(:,7:8),2) mean(phis1(:,num_landmarks+(7:8)),2)];
    reye = [mean(phis1(:,9:10),2) mean(phis1(:,num_landmarks+(9:10)),2)];
    distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
        ((reye(:,2)-leye(:,2)).^2));
end
dsAll = sqrt((del(:,1:num_landmarks,:).^2) + (del(:,num_landmarks+1:num_landmarks*2,:).^2));
dsAll = dsAll./distPup; 
ds=mean(dsAll,2);%        2*sum(dsAll,2)/R;
end

% Just compute mean dis, #Landmarks can be arbitrary
% phis0 and phis1 can be of form shape or phi
function [ds,dsAll] = dist_partial( model, phis0, phis1 )

assert( size(phis0,1) == size(phis1,1) );
assert( size(phis0,2) == size(phis1,2) );

if size(phis0,1) == 2
    num_landmarks = size(phis0,2);
    % Transform to phi form
    phis0 = [ phis0(1,:) phis0(2,:) ];
    phis1 = [ phis1(1,:) phis1(2,:) ];
elseif size(phis0,1) == 1
    num_landmarks = size(phis0,2)/2;
else
    error('invalid option');
end
    
% Compute dis
del = abs( phis0 - phis1 );

dsAll = sqrt((del(:,1:num_landmarks,:).^2) + (del(:,num_landmarks+1:num_landmarks*2,:).^2));

ds_sample =mean(dsAll,2);     %        2*sum(dsAll,2)/R;
ds = mean(ds_sample);
end

% Not only return the mean NME, but also return the NME of each landmark
% ( phis1 = gt )
function [ds,dsAll,ds_per_landmark] = dist_per_landmark( model, phis0, phis1 )

[N,R,T]=size(phis0); 
% dif: just phis0 - phis1( phis1 = gt )
del = dif(phis0,phis1);
if(strcmp(model.name,'cofw'))
    num_landmarks = size(phis1,2)/2;
else
    num_landmarks = size(phis1,2)/2;
end

%Distance between pupils
if(strcmp(model.name,'lfpw') || strcmp(model.name,'cofw'))
    distPup=sqrt(((phis1(:,17)-phis1(:,18)).^2) + ...
                 ((phis1(:,17+num_landmarks)-phis1(:,18+num_landmarks)).^2));
    distPup = repmat(distPup,[1,num_landmarks,T]);
elseif(strcmp(model.name,'lfw'))
    leyeX=mean(phis1(:,1:2),2);leyeY=mean(phis1(:,(1:2)+num_landmarks),2);
    reyeX=mean(phis1(:,7:8),2);reyeY=mean(phis1(:,(7:8)+num_landmarks),2);
    distPup=sqrt(((leyeX-reyeX).^2) + ((leyeY-reyeY).^2));
    distPup = repmat(distPup,[1,num_landmarks,T]);
elseif(strcmp(model.name,'helen'))
    leye = [mean(phis1(:,135:154),2) mean(phis1(:,num_landmarks+(135:154)),2)];
    reye = [mean(phis1(:,115:134),2) mean(phis1(:,num_landmarks+(115:134)),2)];
    distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
        ((reye(:,2)-leye(:,2)).^2));
    distPup = repmat(distPup,[1,num_landmarks,T]);
elseif(strcmp(model.name,'pie'))
    leye = [mean(phis1(:,37:42),2) mean(phis1(:,num_landmarks+(37:42)),2)];
    reye = [mean(phis1(:,43:48),2) mean(phis1(:,num_landmarks+(43:48)),2)];
    distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
        ((reye(:,2)-leye(:,2)).^2));
    distPup = repmat(distPup,[1,num_landmarks,T]);
elseif(strcmp(model.name,'apf'))
    leye = [mean(phis1(:,7:8),2) mean(phis1(:,num_landmarks+(7:8)),2)];
    reye = [mean(phis1(:,9:10),2) mean(phis1(:,num_landmarks+(9:10)),2)];
    distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
        ((reye(:,2)-leye(:,2)).^2));
end
dsAll = sqrt((del(:,1:num_landmarks,:).^2) + (del(:,num_landmarks+1:num_landmarks*2,:).^2));
dsAll = dsAll./distPup; 
ds_per_landmark = mean( dsAll );
ds=mean(dsAll,2);%        2*sum(dsAll,2)/R;
end

% Compute phi that minimizes sum of distances to phis (average shape)
% phis_norm_wrt_bbox: Let the center of bbox ( 0 , 0 )
%                     (Xi - center_x_bbox)/half_width_bbox
%                     (Yi - center_y_bbox)/half_height_bbox
% phiStart: mean phis_norm_wrt_bbox of all N images Format: 1 * (2*NUM_LANDMARKS)
% distPup: dist of pupils for every image ( IOD ) , here , IOD = 55 pixels 
% sz : 1/2 * size of all images 
% bboxes : for COFW , this is known , in other cases , this is calculated.
function [phiStart,phis_norm_wrt_bbox,distPup,sz,bboxes] = compPhiStar( model, phis, Is, pad, imgIds, bboxes )
[N,dimension] = size(phis);
% sz( i , : ) : half the size of image i
sz=zeros(N,2);
if(isempty(imgIds)),imgIds=1:N; end

if(strcmp(model.name,'cofw')), num_landmarks = (dimension/2);
else num_landmarks=dimension/2;
end

phis_norm_wrt_bbox=zeros(N,dimension);
% %%%%%%%%%%%%%%%%% Calculate dist_Pupils for every image( Here , 55 pixels
% for every image , in other words, IOD = 55 for every image )
if(strcmp(model.name,'lfpw') || strcmp(model.name,'cofw'))
    % distPup: distance of pupils( two eyes ), here , distPup = 55 pixels
    distPup=sqrt(((phis(:,17)-phis(:,18)).^2)+((phis(:,17+num_landmarks)-phis(:,18+num_landmarks)).^2));
elseif(strcmp(model.name,'mouseP'))
    distPup=68;
elseif(strcmp(model.name,'lfw'))
    leyeX=mean(phis(:,1:2),2);leyeY=mean(phis(:,(1:2)+num_landmarks),2);
    reyeX=mean(phis(:,7:8),2);reyeY=mean(phis(:,(7:8)+num_landmarks),2);
    distPup=sqrt(((leyeX-reyeX).^2) + ((leyeY-reyeY).^2));
else distPup=0;
end

if(nargin<6), bboxes = zeros(N,4); end
for n=1:N
    if(nargin<6)
        % left top width height
        % design a bbox to contain all ground_truth points
        % reserve a padding : pad = 10 
        % left:pad pixels right:2*pad pixels
        % up: pad pixels down: 2*pad pixels
        bboxes(n,1) = min(phis(n,1:num_landmarks)) - pad ;                                  % Left
        bboxes(n,2) = min(phis(n,num_landmarks+1:end)) - pad;                               % Upper
        bboxes(n,3) = max(phis(n,1:num_landmarks))-bboxes(n,1) + 2*pad;                     % Width
        bboxes(n,4) = max(phis(n,num_landmarks+1:num_landmarks*2)) -bboxes(n,2) + 2*pad;    % Height
    end
    img = Is{imgIds(n)}; 
    [sz(n,1),sz(n,2),~]=size(img);
    sz(n,:)=sz(n,:)/2;
    %All relative to centroid, using bbox size
    if(nargin<6)
        szX=bboxes(n,3)/2;
        szY=bboxes(n,4)/2;
        center_bbox(1)=bboxes(n,1)+szX;             
        center_bbox(2) = bboxes(n,2)+szY;
    else
        szX=bboxes(imgIds(n),3)/2;
        szY=bboxes(imgIds(n),4)/2;
        % center_bbox: center_x center_y
        center_bbox(1)=bboxes(imgIds(n),1)+szX;     
        center_bbox(2) = bboxes(imgIds(n),2)+szY;
    end
    % szX : 1/2 * width_of_bbox
    % szY : 1/2 * height_of_bbox
    if(  strcmp(model.name,'cofw')  )
        phis_norm_wrt_bbox(n,:)  = [( phis( n , 1:num_landmarks)-center_bbox(1))./szX ... 
                      ( phis( n , num_landmarks+1:num_landmarks*2)-center_bbox(2))./szY ...
                        phis( n , (num_landmarks*2)+1:num_landmarks*3)];
    else
        phis_norm_wrt_bbox(n,:) = [ (phis( n,1:num_landmarks)-center_bbox(1)) ./ szX ...
                       (phis( n,num_landmarks+1:num_landmarks*2)-center_bbox(2)) ./ szY];
    end
end
% phiStart is the mean phis_norm_wrt_bbox of all training shapes.
phiStart = mean(phis_norm_wrt_bbox,1);
end

% %%%%%%%%% Initialize Training Settings %%%%%%%%%%%
function [ pCur, phi_true, phi_true_norm_wrt_bbox,pStart,imgIds,N,N1] = initTr( Is,phi_true,...
    model,pStart,bboxes,L,pad)
%Randomly initialize each training image with L shapes
[N,dimension]=size(phi_true);
assert(length(Is)==N);
% If pStart is not given, we calculate the mean phi_true_norm_wrt_bbox as
% pStart
if(isempty(pStart)),
    [pStart,phi_true_norm_wrt_bbox]=compPhiStar(model,phi_true,Is,pad,[],bboxes);
end
% augment data amount by random permutations of initial shape
pCur=repmat(phi_true,[1,1,L]);
if(strcmp(model.name,'cofw'))
    num_landmarks = size(phi_true,2)/3;
else num_landmarks = size(phi_true,2)/2;
end
for n=1:N
    %select other images
    imgsIds = randSample([1:n-1 n+1:N],L);%[n randSample(1:N,L-1)];
    %Project onto image
    for i=1:L
        %permute bbox location slightly
        maxDisp = bboxes(n,3:4)/16;
        uncert=(2*rand(1,2)-1).*maxDisp;
        bbox=bboxes(n,:);
        bbox(1:2) = bbox(1:2) + uncert;
        % reprojectPose: Given the phi_true_norm_wrt_bbox and bbox
        %                Output: raw image coordinates of Phi
        pCur(n,:,i) = reprojectPose( model , phi_true_norm_wrt_bbox(imgsIds(i),:) , bbox);
    end
end
% Rearrange pCur
% before this operation, pCur : N * (num_landmarks*2/3) * L
% after this operation, pCur : (N*L) * (num_landmarks*2/3)
if(strcmp(model.name,'cofw'))
    pCur = reshape(permute(pCur,[1 3 2]),N*L,num_landmarks*3);
else
    pCur = reshape(permute(pCur,[1 3 2]),N*L,num_landmarks*2);
end
% imgIds: 1 2 ... 1345 1 2 ... 1345 1 2 ... 1345 ... 1 2 ...1345( L times )
imgIds =repmat( 1:N,[1 L]);
phi_true =repmat( phi_true, [L 1] );
phi_true_norm_wrt_bbox = repmat(phi_true_norm_wrt_bbox,[L 1]);
N1=N; 
N=N*L;
end

function p=initTest(Is,bboxes,model,pStart,phi_true_norm_wrt_bbox,RT1)
%Randomly initialize testing shapes using training shapes (RT1 different)
N=length(Is);dimension=size(pStart,2);phis_norm_wrt_bbox=phi_true_norm_wrt_bbox;
if(isempty(bboxes)), p=pStart(ones(N,1),:);
    %One bbox provided per image
elseif(ismatrix(bboxes) && size(bboxes,2)==4),
    p=zeros(N,dimension,RT1);NTr=size(phis_norm_wrt_bbox,1);%gt=regModel.phisT;
    for n=1:N
        %select other images
        imgsIds = randSample(NTr,RT1);
        %Project into image
        for l=1:RT1
            %permute bbox location slightly (not scale)
            maxDisp = bboxes(n,3:4)/16;
            uncert=(2*rand(1,2)-1).*maxDisp;
            bbox=bboxes(n,:);bbox(1:2)=bbox(1:2)+uncert;
            p(n,:,l)=reprojectPose(model,phis_norm_wrt_bbox(imgsIds(l),:),bbox);
        end
    end
    %RT1 bboxes given, just reproject
elseif(size(bboxes,2)==4 && size(bboxes,3)==RT1)
    p=zeros(N,dimension,RT1);NTr=size(phis_norm_wrt_bbox,1);
    for n=1:N
        imgsIds = randSample(NTr,RT1);
        for l=1:RT1
            p(n,:,l)=reprojectPose(model,phis_norm_wrt_bbox(imgsIds(l),:),...
                bboxes(n,:,l));
        end
    end
    %Previous results are given, use as is 
elseif(size(bboxes,2)==dimension && size(bboxes,3)==RT1)
    p=bboxes;
    %VIDEO
elseif(iscell(bboxes))
    p=zeros(N,dimension,RT1);NTr=size(phi_true_norm_wrt_bbox,1);
    for n=1:N
        bb=bboxes{n}; ndet=size(bb,1);
        imgsIds = randSample(NTr,RT1);
        if(ndet<RT1), bbsIds=randint2(1,RT1,[1,ndet]);
        else bbsIds=1:RT1;
        end
        for l=1:RT1
            p(n,:,l)=reprojectPose(model,phi_true_norm_wrt_bbox(imgsIds(l),:),...
                bb(bbsIds(l),1:4));
        end
    end
end
end

% Input:
% diff = cur_pose - model_poses
% format:
% del_x1 del_x2 del_x3 ... del_xn
% del_y1 del_y2 del_y3 ... del_yn
% dist_type can be 'L1' or 'L2'
% Output:
% L1/L2 distance 
% format:
% L1_dis_1 L1_dis_2 L1_dis_3 ... L1_dis_n
function mag_dis = dist_pose( diff , dist_type )
    % column vector
    assert( size( diff , 1 ) == 2 );
    
    if strcmp( dist_type , 'L1' )
        mag_dis = sum( abs( diff ) );   
    elseif strcmp( dist_type , 'L2' )
        mag_dis = sqrt( sum( diff.^2 ) );
    else
        error( 'ShapeGt dist_pose() dist_type error' );
    end
end

% phi : current phi. Format:x1 x2 ... xn y1 y2 ... yn. Image coords.
% para : string. 'hog' or other kinds of features
% Output: feature col vector
% [f1_1 f1_2 f1_3 ... f1_36 f2_1 f2_2 ... f2_36 ... f68_36 ]'
function [ feature ] = extract_feature_MATLAB( img , phi , para )
    global NUM_LANDMARKS;
    global feat_para;
    global cnt_out_of_range;    % count how many points are out ot range when the features are extracted.
    debug = 1;

    xs = phi( 1 : NUM_LANDMARKS );
    ys = phi( NUM_LANDMARKS+1 : 2*NUM_LANDMARKS );
    coords = [ xs ; ys ];
    w = size( img , 2 );
    h = size( img , 1 );
    assert( w > 0 && h > 0 );
    debug = 0;
   
    if strcmp( para.feature , 'hog' )
        % %%%%%%  Set parameters of HOG %%%%%%
        CellSize   = feat_para.CellSize;
        BlockSize  = feat_para.BlockSize;
        NumBins    = feat_para.NumBins;
        UseSignedOrientation = feat_para.UseSignedOrientation;
        Size_Feat  = feat_para.Size_Feat;             % size of feature for one point
        if isfield( para , 'pad_mode' )
            PAD_MODE = para.pad_mode;
        else
            PAD_MODE = 'replicate';
        end
        
        % PAD = CellSize + 2
        PAD = max( CellSize ) * max(BlockSize)/2 + 3;
        %%%%%%%  Some points are close to boundaries. Just replicate the border pixels !!! %%%%%%
        x_min = min( xs ); x_max = max( xs );
        y_min = min( ys ); y_max = max( ys );
        if x_min < PAD  % pad left
            if debug == 1
                warning('Out of range');
            end
            cnt_out_of_range = cnt_out_of_range +1;
            % Adjust the coords
            if x_min < 1
               xs( xs < 1 ) = 1 ; 
            end
            img = padarray( img , [ 0 , PAD , 0 ] , 'pre' , PAD_MODE );
            % Adjust the coords
            xs = xs + PAD;
            coords = [ xs ; ys ];
            % debug
            if debug > 0
                disp('x min out of range');
            end
        end
        if x_max + PAD >= w  % pad right
            if debug == 1
                warning('Out of range');
            end
            cnt_out_of_range = cnt_out_of_range +1;
            % Let all ys( ys < 1 ) = 1
            if x_max >= w
               coords( 1 , coords( 1 , :) >= w ) = w ; 
            end
            img = padarray( img , [ 0 , PAD , 0 ] , 'post' , PAD_MODE );
            if debug > 0
                disp('x max out of range');
            end
        end
        if y_min < PAD  % pad up
            if debug == 1
                warning('Out of range');
            end
            cnt_out_of_range = cnt_out_of_range +1;
            % Adjust the coords
            % Let all ys( ys < 1 ) = 1
            if y_min < 1
               coords( 2 , coords( 2 , :) < 1 ) = 1 ; 
            end
            img = padarray( img , [ PAD , 0 , 0 ] , 'pre' , PAD_MODE );
            % Adjust the coords
            coords( 2 , : ) = coords( 2 , : ) + PAD;
            if debug > 0
                disp('y min out of range');
            end 
        end
        if y_max + PAD >= h  % pad down
            if debug == 1
                warning('Out of range');
            end
            cnt_out_of_range = cnt_out_of_range +1;
            if y_max >= h
               coords( 2 , coords( 2 , :) >= h ) = h ; 
            end
            img = padarray( img , [ PAD , 0 , 0 ] , 'post' , PAD_MODE );
            if debug > 0
                disp('y max out of range');
            end
        end
    end
    
    % Crop the region to speed up HOG Feature Extraction .
    left_bound   = round( min( coords(1,:) ) ) - (PAD-1);
    right_bound  = round( max( coords(1,:) ) ) + (PAD-1);
    upper_bound  = round( min( coords(2,:) ) ) - (PAD-1);
    bottom_bound = round( max( coords(2,:) ) ) + (PAD-1);

    Sub_Image = img( upper_bound : bottom_bound , left_bound : right_bound , : );
    % Adjust coords
    coords(2,:) = coords(2,:) - ( upper_bound - 1 );
    coords(1,:) = coords(1,:) - ( left_bound - 1 );
    
    %%%%%%%%%%%%%%%%%%%% Hog Feature Extraction %%%%%%%%%%%%%%%%%%%%
    % format of feature: every row for one point
    [ feature , validPoints ] = extractHOGFeatures( Sub_Image , coords' , ...
                                  'CellSize' , CellSize , ...
                                  'BlockSize', BlockSize , ...
                                  'NumBins' ,  NumBins , ...
                                  'UseSignedOrientation' , UseSignedOrientation ... 
                                );
    assert( size(validPoints,1) == size(coords,2) );
    %%% Concatenate feature into a column feature vector
    feature = reshape( feature' , NUM_LANDMARKS * Size_Feat , 1 );
    
    %%%%%%%%%%%%%%%%%%%% Multi-scale HOG Feature %%%%%%%%%%%%%%%%%%%%
    if isfield( feat_para , 'ScaleRatio' )
        rate1 = 0.8;
        Sub_Image = imresize( Sub_Image , rate1 );
        coords = coords * rate1;
        multi_scale_feature = extractHOGFeatures( Sub_Image , coords' , ...
                                    'CellSize' , CellSize , ...
                                    'BlockSize', [2 2] , ...
                                    'NumBins' ,  NumBins , ...
                                    'UseSignedOrientation' , UseSignedOrientation ... 
                                    );
        BlockSize = max( BlockSize );
        Size_Feat = NumBins * 2 * 2;
        multi_scale_feature = reshape( multi_scale_feature' , NUM_LANDMARKS * Size_Feat , 1 );
        feature = [ feature ; multi_scale_feature ];
        
        rate2 = rate1 * rate1;
        Sub_Image = imresize( Sub_Image , rate2 );
        coords = coords * rate2;
        multi_scale_feature = extractHOGFeatures( Sub_Image , coords' , ...
                                    'CellSize' , CellSize , ...
                                    'BlockSize', [2 2] , ...
                                    'NumBins' ,  NumBins , ...
                                    'UseSignedOrientation' , UseSignedOrientation ... 
                                    );
        BlockSize = max( BlockSize );
        Size_Feat = NumBins * 2 * 2;
        multi_scale_feature = reshape( multi_scale_feature' , NUM_LANDMARKS * Size_Feat , 1 );
        feature = [ feature ; multi_scale_feature ];
    end
end

% Input : 
% img : rows * cols * channels unit8 or single
% phi : current phi. Format:x1 x2 ... xn y1 y2 ... yn. Image coords.
% para : string. 'hog' or other kinds of features
% Output: feature col vector
% [f1_1 f1_2 f1_3 ... f1_36 f2_1 f2_2 ... f2_36 ... f68_36 ]'
function [ feature ] = extract_feature( img , phi , para )
    global NUM_LANDMARKS;
    global feat_para;
    global cnt_out_of_range;    % count how many points are out ot range when the features are extracted.
    debug = 1;

    xs = phi( 1 : NUM_LANDMARKS );
    ys = phi( NUM_LANDMARKS+1 : 2*NUM_LANDMARKS );
    coords = [ xs ; ys ];
    w = size( img , 2 );
    h = size( img , 1 );
    assert( w > 0 && h > 0 );
    debug = 0;
   
    if strcmp( para.feature , 'hog' )
        % %%%%%%  Set parameters of HOG %%%%%%
        CellSize   = feat_para.CellSize;
        BlockSize  = feat_para.BlockSize;
        NumBins    = feat_para.NumBins;
        UseSignedOrientation = feat_para.UseSignedOrientation;
        Size_Feat  = feat_para.Size_Feat;             % size of feature for one point
        if isfield( para , 'pad_mode' )
            PAD_MODE = para.pad_mode;
        else
            PAD_MODE = 'replicate';
        end
        
        % PAD = CellSize + 2
        PAD = max( CellSize ) * max(BlockSize)/2 + 3;
        %%%%%%%  Some points are close to boundaries. Just replicate the border pixels !!! %%%%%%
        x_min = min( xs ); x_max = max( xs );
        y_min = min( ys ); y_max = max( ys );
        if x_min < PAD  % pad left
            if debug == 1
                warning('Out of range');
            end
            cnt_out_of_range = cnt_out_of_range +1;
            % Adjust the coords
            if x_min < 1
               xs( xs < 1 ) = 1 ; 
            end
            img = padarray( img , [ 0 , PAD ] , 'pre' , PAD_MODE );
            % Adjust the coords
            xs = xs + PAD;
            coords = [ xs ; ys ];
            % debug
            if debug > 0
                disp('x min out of range');
            end
        end
        if x_max + PAD >= w  % pad right
            if debug == 1
                warning('Out of range');
            end
            cnt_out_of_range = cnt_out_of_range +1;
            % Let all ys( ys < 1 ) = 1
            if x_max >= w
               coords( 1 , coords( 1 , :) >= w ) = w ; 
            end
            img = padarray( img , [ 0 , PAD ] , 'post' , PAD_MODE );
            if debug > 0
                disp('x max out of range');
            end
        end
        if y_min < PAD  % pad up
            if debug == 1
                warning('Out of range');
            end
            cnt_out_of_range = cnt_out_of_range +1;
            % Adjust the coords
            % Let all ys( ys < 1 ) = 1
            if y_min < 1
               coords( 2 , coords( 2 , :) < 1 ) = 1 ; 
            end
            img = padarray( img , [ PAD , 0 ] , 'pre' , PAD_MODE );
            % Adjust the coords
            coords( 2 , : ) = coords( 2 , : ) + PAD;
            if debug > 0
                disp('y min out of range');
            end 
        end
        if y_max + PAD >= h  % pad down
            if debug == 1
                warning('Out of range');
            end
            cnt_out_of_range = cnt_out_of_range +1;
            if y_max >= h
               coords( 2 , coords( 2 , :) >= h ) = h ; 
            end
            img = padarray( img , [ PAD , 0 ] , 'post' , PAD_MODE );
            if debug > 0
                disp('y max out of range');
            end
        end
    end
    
    % Crop the region to speed up HOG Feature Extraction .
    left_bound   = round( min( coords(1,:) ) ) - (PAD-1);
    right_bound  = round( max( coords(1,:) ) ) + (PAD-1);
    upper_bound  = round( min( coords(2,:) ) ) - (PAD-1);
    bottom_bound = round( max( coords(2,:) ) ) + (PAD-1);
    Sub_Image = img( upper_bound : bottom_bound , left_bound : right_bound );
    % Adjust coords
    coords(2,:) = coords(2,:) - ( upper_bound - 1 );
    coords(1,:) = coords(1,:) - ( left_bound - 1 );
    
    %%%%%%%%%%%%%%%%%%%% Hog Feature Extraction %%%%%%%%%%%%%%%%%%%%
    [ M , O ] = gradientMag( single(Sub_Image) );
    PARAMS = struct( ...
                    'CellSize' , max(CellSize) , ...
                    'BlockSize', max(BlockSize) , ...
                    'NumBins' ,  NumBins , ...
                    'UseSignedOrientation' , UseSignedOrientation ...
                   );
    % format of feature: every row for one point
    feature = myHog( double(M) , double(O) , coords' , PARAMS );
 
    %%%%%%%%%%%%%%%%%%%% Concatenate feature into a column feature vector  %%%%%%%%%%%%%%%%%%%%
    feature = reshape( feature' , NUM_LANDMARKS * Size_Feat , 1 );
    
end

% shape: can be of format: 
% shape:   [ x1 x2 ... xn ; y1 y2 ... yn ] OR 
% atom:    [x1 y1 x2 y2... xn yn]' OR 
% phi:     [ x1 x2 ... xn y1 y2 ... yn ]
% params: string, 'shape' or 'atom' or 'phi'
function output = disp_shape( shape , params )
   global NUM_LANDMARKS;
   assert( ~isempty(NUM_LANDMARKS) && NUM_LANDMARKS >0 ); 
   if strcmp( params , 'shape' )
       xs = shape(1,:);
       ys = shape(2,:);
   elseif strcmp( params , 'atom' )
       shape = reshape( shape , 2 , NUM_LANDMARKS );
       xs = shape(1,:);
       ys = shape(2,:);
   elseif strcmp( params , 'phi' )
       xs = shape(1:NUM_LANDMARKS);
       ys = shape(NUM_LANDMARKS+1 : 2*NUM_LANDMARKS);
   else
       error('disp_shape: params error');
   end
   
   % visualize the results
   sz = 40;
   scatter(xs, -ys, sz, 'MarkerEdgeColor',[0 .5 .5],...
                        'MarkerFaceColor',[0 .7 .7],...
                        'LineWidth',1.5);
   %axis off;
   output = 0;
end

% id: the id-th image in train_set
function output = disp_keypoints( id )
    global imgs_train;
    global imgs_attri_train;
    img = imgs_train{id};
    keypoints = imgs_attri_train(id).norm_coord_just_keypoints ;
    
    disp_landmarks( img , keypoints );
    
    output = 0;
end

% phi : of format: [x1 x2 ... xn y1 y2 ... yn ]
function IOD = computeIOD_phi( phi )
    global NUM_LANDMARKS;
    assert( size(phi,1) == 1 && size(phi,2) == 2*NUM_LANDMARKS );
    shape = [ phi(1:NUM_LANDMARKS); phi(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
    if NUM_LANDMARKS == 68
        left_eye_2D    =  mean( [ shape(:,37) shape(:,38) shape(:,39) shape(:,40) shape(:,41) shape(:,42) ] , 2) ;
        right_eye_2D   =  mean( [ shape(:,43) shape(:,44) shape(:,45) shape(:,46) shape(:,47) shape(:,48) ] , 2) ;
    elseif NUM_LANDMARKS == 29
        left_eye_2D = shape(:,17);
        right_eye_2D = shape(:,18);
    else
        error('Wrong param');
    end
    IOD            =  norm( left_eye_2D - right_eye_2D );
end

% atom : of format: [x1 y1 x2 y2 ... xn yn]'
function IOD = computeIOD_atom( atom )
    global NUM_LANDMARKS;
    assert( size(atom,1) == 2*NUM_LANDMARKS && size(atom,2) == 1 );
    shape = reshape( atom , 2 , NUM_LANDMARKS );
    if NUM_LANDMARKS == 68
        left_eye_2D    =  mean( [ shape(:,37) shape(:,38) shape(:,39) shape(:,40) shape(:,41) shape(:,42) ] , 2) ;
        right_eye_2D   =  mean( [ shape(:,43) shape(:,44) shape(:,45) shape(:,46) shape(:,47) shape(:,48) ] , 2) ;
    elseif NUM_LANDMARKS == 29
        left_eye_2D  = mean( shape(:,[9 11 13 14]) , 2 );
        right_eye_2D = mean( shape(:,[10 12 15 16]) ,2 );
    else
        error('Wrong param');
    end
    IOD            =  norm( left_eye_2D - right_eye_2D );
end

% shape : of format: [x1 x2 x3...xn ; y1 y2 y3 ... yn ]
function IOD = computeIOD_shape( shape )
    global NUM_LANDMARKS;
    assert( size(shape,1) == 2 && size(shape,2) == NUM_LANDMARKS );
    if NUM_LANDMARKS == 68
        left_eye_2D    =  mean( [ shape(:,37) shape(:,38) shape(:,39) shape(:,40) shape(:,41) shape(:,42) ] , 2) ;
        right_eye_2D   =  mean( [ shape(:,43) shape(:,44) shape(:,45) shape(:,46) shape(:,47) shape(:,48) ] , 2) ;
    elseif NUM_LANDMARKS == 29
        left_eye_2D = shape(:,17);
        right_eye_2D = shape(:,18);
    else
        error('Wrong param');
    end
    IOD            =  norm( left_eye_2D - right_eye_2D );
end

% A wrapper for computeIOD_shape , computeIOD_atom and computeIOD_phi
function IOD = computeIOD( shape , params)
    global NUM_LANDMARKS;
    IOD = 0;
    if strcmp( params , 'phi' )
        IOD = computeIOD_phi( shape );
    elseif strcmp( params , 'shape' )
        IOD = computeIOD_shape( shape );
    elseif strcmp( params , 'atom' )
        IOD = computeIOD_atom( shape );
    else
        error( 'params invalid.' );
    end
end

% Compute delta_S from normalized_wrt_bbox to raw image coords
% Input:
% delta_S : delta_S normalized_wrt_bbox 
% bboxes : bboxes
% Output:
% delta_S_img_coords
function delta_S_img_coords = delta_norm2raw( delta_S , bboxes )
    global NUM_LANDMARKS;
    delta_S_img_coords( : , 1:NUM_LANDMARKS ) = delta_S( : , 1:NUM_LANDMARKS ) .* bboxes(:,3);
    delta_S_img_coords( : , NUM_LANDMARKS+1 : 2*NUM_LANDMARKS ) = delta_S( : , NUM_LANDMARKS+1 : 2*NUM_LANDMARKS ) .* bboxes(:,4);
end

% Flip coords in image I
% bbox: [ left , upper , width , height ]
% size_img: size(I) , of format( height , width )
function output_coords = flip_bbox( bbox , size_img )
    % check
    assert( size(bbox,1) == 1 & size(bbox,2) == 4 );
    assert( size(size_img,1) == 1 & size(size_img,2) == 2 );
    
    h = size_img(1);
    w = size_img(2);
    left_upper = bbox(1:2);
    right_down = bbox(1:2) + bbox(3:4);
    % flip left_upper and right_down point
    left_upper(1) = w + 1 - left_upper(1);
    right_down(1) = w + 1 - right_down(1);
    
    % generate output bbox
    output_coords = [ right_down(1) left_upper(2) bbox(3) bbox(4) ];
end

% box : [ left upper width height ]
% size_img: size(I) , of format( height , width )
function new_box = flip_box( box , size_img )
    % check parameters
    global NUM_LANDMARKS;
    if isempty( box )
        new_box = [];
        return;
    end
    assert( size( box , 1 ) == 1 && size( box , 2 ) == 4 && size(size_img,1) == 1 && size(size_img,2) == 2 );
    center = [ box(1) + box(3)/2; box(2) + box(4)/2 ];
    width = size_img(2);
    center(1) = width + 1 - center(1);
    new_box = [ center(1) - box(3)/2 , center(2) - box(4)/2 , box(3) , box(4) ];
end

% Flip coords in image I
% coords: can be [ x1 x2 ... xn ;
%                  y1 y2 ... yn ]
%         or [ occlu_label_1 ol2 ol3 ... ol_n ]
% size_img: size(I) , of format( height , width )
function output_coords = flip_coords( coords , size_img )
    % check parameters
    global NUM_LANDMARKS;
    assert( ~isempty(NUM_LANDMARKS) );
    assert( size( coords , 1 ) == 2  && size(size_img,1) == 1 && size(size_img,2) == 2 );
    width = size_img(2);
    coords(1,:) = width + 1 - coords(1,:);
    output_coords = coords;
    
    % Adjust the index !  For convenience, manually !
    if NUM_LANDMARKS == 68 && size( coords , 2 ) == NUM_LANDMARKS
        swap_pairs = [...
                      1 17;...
                      2 16;...
                      3 15;...
                      4 14;...
                      5 13;...
                      6 12;...
                      7 11;...
                      8 10;...
                      18 27;...
                      19 26;...
                      20 25;...
                      21 24;...
                      22 23;...
                      37 46;...
                      38 45;...
                      39 44;...
                      40 43;...
                      41 48;...
                      42 47;...
                      32 36;...
                      33 35;...
                      49 55;...
                      61 65;...
                      50 54;...
                      51 53;...
                      62 64;...
                      68 66;...
                      59 57;...
                      60 56 ];
    elseif NUM_LANDMARKS == 29 && size( coords , 2 ) == NUM_LANDMARKS
        swap_pairs = [...
                      1 2;...
                      3 4;...
                      5 7;...
                      6 8;...
                      9 10;...
                      11 12;...
                      13 15;...
                      17 18;...
                      14 16;...
                      19 20;...
                      23 24;...
                      ];
    elseif size( coords , 2 ) == 5
        % order of coords: left_eye right_eye nose left_mouth right_mouth
        swap_pairs = [...
                      1 2;...
                      4 5; ];
    else
        error( 'invalid parameters' );
    end
    % swap the coords
    num = size( swap_pairs , 1 );
    for i = 1 : num
        left  = swap_pairs( i , 1 );
        right = swap_pairs( i , 2 );
        % swapt the coords
        t = output_coords( : , left );
        output_coords( : , left ) = output_coords( : , right );
        output_coords( : , right ) = t;
    end
end

% Flip occlusion labels in image I
% labels: [ occlu_label_1 ol2 ol3 ... ol_n ]
function output_labels = flip_labels( labels  )
    global NUM_LANDMARKS;
    assert( size(labels,1)==1 && size(labels,2)==NUM_LANDMARKS);
    output_labels = labels;
    if NUM_LANDMARKS == 29 
        swap_pairs = [...
                      1 2;...
                      3 4;...
                      5 7;...
                      6 8;...
                      9 10;...
                      11 12;...
                      13 15;...
                      17 18;...
                      14 16;...
                      19 20;...
                      23 24;...
                      ];
    elseif NUM_LANDMARKS == 68
        swap_pairs = [...
                      1 17;...
                      2 16;...
                      3 15;...
                      4 14;...
                      5 13;...
                      6 12;...
                      7 11;...
                      8 10;...
                      18 27;...
                      19 26;...
                      20 25;...
                      21 24;...
                      22 23;...
                      37 46;...
                      38 45;...
                      39 44;...
                      40 43;...
                      41 48;...
                      42 47;...
                      32 36;...
                      33 35;...
                      49 55;...
                      61 65;...
                      50 54;...
                      51 53;...
                      62 64;...
                      68 66;...
                      59 57;...
                      60 56 ];
    else
        error('remains to be developed');
    end
    % swap the labels
    num = size( swap_pairs , 1 );
    for i = 1 : num
        left  = swap_pairs( i , 1 );
        right = swap_pairs( i , 2 );
        % swapt the coords
        t = output_labels( : , left );
        output_labels( : , left ) = output_labels( : , right );
        output_labels( : , right ) = t;
    end
end

% normally distributed random number between [0 1]
function rs = randn_01( sz )
    % size
    assert( size(sz,1) > 0 && size(sz,2) > 0 );
    r = randn( sz );
    max_value = max( max( abs( r ) ) );
    rs = r / max_value ;
end

% Compute mean NME of estimate wrt ground truth
% truth : ground truth , matrix , one column is one shape.
% estimate : estimate , matrix , one column is one shape.
% params: can be 'atom' or 'phi'
%         'atom' : each column is an atom shape
%         'phi'  : each row is an phi shape
function [ mean_NME , NMEs ] = computeNME( truth , estimate , params )
    global NUM_LANDMARKS;
    assert( size(truth,1) == size(estimate,1) && size(truth,2) == size(estimate,2) );
    assert( ischar(params) );
    
    if strcmp( params , 'atom' ) 
        difs = truth - estimate;
        % Computes NME
        NMEs = zeros( size(truth,2) , 1 );
        for i = 1 : size(truth,2)
            shape = truth(:,i);
            dif = difs(:,i);
            shape = reshape( shape , 2 , NUM_LANDMARKS );
            dif = reshape( dif , 2 , NUM_LANDMARKS );
            % compute IOD
            IOD = computeIOD( shape , 'shape' );
            NMEs(i) = mean( sqrt( sum( dif.^2 ) ) ) / IOD ;
        end

        mean_NME = mean( NMEs );
    elseif strcmp( params , 'phi' )
        error('remains to be developed');
    else
        error('invalid params');
    end
end

% change the format of shape 
% phi to shape / atom
% shape to atom / phi
% atom to shape / phi
% format: target form
function output = changeShape( input , format )
    global NUM_LANDMARKS;
    if strcmp( format , 'phi' )
        if size( input , 1 ) == 2 && size(input,2) == NUM_LANDMARKS % shape
            output = [ input(1,:) input(2,:) ];
        elseif size(input,1) == 2*NUM_LANDMARKS && size(input,2) == 1 % atom
            output = reshape( input , 2 , NUM_LANDMARKS);
            output = [ output(1,:) output(2,:) ];
        else
            error('invalid param');
        end
    elseif strcmp( format , 'shape' )
        if size(input,1) == 2*NUM_LANDMARKS && size(input,2) == 1 % atom
            output = reshape( input , 2 , NUM_LANDMARKS );
        elseif size(input,1) == 1 && size(input,2) == 2*NUM_LANDMARKS %phi
            output = [ input(1:NUM_LANDMARKS);input(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
        else
            error('invalid param');
        end
    elseif strcmp( format , 'atom' )
        if size( input , 1 ) == 2 && size(input,2) == NUM_LANDMARKS % shape
            output = reshape( input , 2*NUM_LANDMARKS , 1 );
        elseif size(input,1) == 1 && size(input,2) == 2*NUM_LANDMARKS %phi
            output = [ input(1:NUM_LANDMARKS);input(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
            output = reshape( output , 2*NUM_LANDMARKS , 1 );
        else
            error('invalid param');
        end
    else
        error('invalid param');
    end
end

% Two normalization steps:
% 1. Centerization: Let train_para.center_shape be the center of a shape
% 2. Scaling      : norm( shape ) = 1
% Parameters:
% pTrue:  every row is a phi 
%%%%%%%%%%
% Should also consider transforming back !!!!!!!So,store transforming info
function rs = normalize_shape( pTrue )
    global train_para;
    global NUM_LANDMARKS;
    scale_ratio = 2;
    
    rs.new_shapes = zeros(size(pTrue));
    %%%%% Used for transforming back !
    rs.scalings   = zeros(1,size(pTrue,1));
    rs.centers    = zeros(2,size(pTrue,1));
    
    if strcmp( train_para.center_shape , 'nose_tip' )
        for i = 1 : size(pTrue,1)
            phi = pTrue(i,:);
            shape = [ phi(1:NUM_LANDMARKS); phi(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
            % nose tip index depends on NUM_LANDMARKS = 68 or 29
            if NUM_LANDMARKS == 68
                center = shape(:,31);
            elseif NUM_LANDMARKS == 29
                center = shape(:,21);
            else error('invalid option');end
            % let center be (0,0)
            shape = shape - repmat(center , 1 , NUM_LANDMARKS);
            % scaling the shape
            phi = [ shape(1,:) shape(2,:) ];
            rs.scalings(i)  = norm(phi) / scale_ratio;
            phi = phi / rs.scalings(i);
            pTrue(i,:) = phi;
            % Store transforming info
            rs.centers(:,i) = center;
        end
    elseif strcmp( train_para.center_shape , 'center' )
        for i = 1 : size(pTrue,1)
            phi = pTrue(i,:);
            shape = [ phi(1:NUM_LANDMARKS); phi(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
            % nose tip index depends on NUM_LANDMARKS = 68 or 29
            center = mean( shape , 2 );
            % let center be (0,0)
            shape = shape - repmat(center , 1 , NUM_LANDMARKS);
            % scaling the shape
            phi = [ shape(1,:) shape(2,:) ];
            rs.scalings(i)  = norm(phi) / scale_ratio;
            phi = phi / rs.scalings(i);
            pTrue(i,:) = phi;
            % Store transforming info
            rs.centers(:,i) = center;
        end
    elseif strcmp( train_para.center_shape , 'eyes_center' )
        for i = 1 : size(pTrue,1)
            phi = pTrue(i,:);
            shape = [ phi(1:NUM_LANDMARKS); phi(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
            % nose tip index depends on NUM_LANDMARKS = 68 or 29
            if NUM_LANDMARKS == 68
                center = mean( shape(:,[37:42 43:48]) , 2 );
            elseif NUM_LANDMARKS == 29
                center = mean( shape(:,[9 11 13 14 12 10 15 16 17 18]) , 2 );
            else error('invalid option');end
            % let center be (0,0)
            shape = shape - repmat(center , 1 , NUM_LANDMARKS);
            % scaling the shape
            phi = [ shape(1,:) shape(2,:) ];
            rs.scalings(i)  = norm(phi) / scale_ratio ;
            phi = phi / rs.scalings(i);
            pTrue(i,:) = phi;
            % Store transforming info
            rs.centers(:,i) = center;
        end
    elseif strcmp( train_para.center_shape , '1_28' )
        assert( NUM_LANDMARKS == 29 );
        for i = 1 : size(pTrue,1)
            phi = pTrue(i,:);
            shape = [ phi(1:NUM_LANDMARKS); phi(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
            % nose tip index depends on NUM_LANDMARKS = 68 or 29
            center = mean( shape(:,1:28) , 2 );

            % let center be (0,0)
            shape = shape - repmat(center , 1 , NUM_LANDMARKS);
            % scaling the shape
            phi = [ shape(1,:) shape(2,:) ];
            rs.scalings(i)  = norm(phi) / scale_ratio ;
            phi = phi / rs.scalings(i);
            pTrue(i,:) = phi;
            % Store transforming info
            rs.centers(:,i) = center;
        end
    else error('invalid option');end
    
    rs.new_shapes = pTrue;
end

%%%%%% Align cur shapes to the mean shape 
% phis        : every row is a phi, NOTE: raw image coords,
% mean_shape  : corresponding mean shape , phi form 
% NOTE        : to represent a shape , center of the shape MUST BE CHOSEN !    
% output:
% rs.phis : phis that are aligned to the mean shape
% rs.shape_centers: 
% rs.gamas: gama that are applied to the transformation to mean shape
%           and the shape center
% NOTE: pipeline:
%       Translation -> Gama 
function rs = align_to_mean_shape( phis , mean_shape )
    global train_para;
    global NUM_LANDMARKS;
    
    % check
    assert( size(phis,2) == 2*NUM_LANDMARKS & size(mean_shape,2) == 2*NUM_LANDMARKS );
    assert( max(max(abs(phis))) > 1 & max(max(abs(mean_shape))) < 1 );
    r = normalize_shape( mean_shape );
    mean_shape = r.new_shapes;
    
    meanShape = [ mean_shape(1:NUM_LANDMARKS); mean_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
    
    %%%%% Procrustes Analysis to mean shape && Centerize the shape
    rs.phis  = zeros( size(phis,1) , size(phis,2) );
    for i = 1 : size(phis,1)
        phi = phis(i,:);
        shape = [ phi(1:NUM_LANDMARKS); phi(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
        % To make the algorithm more robust, centerization is needed
        center_shape = mean( shape , 2 );
        shape = shape - repmat( center_shape , 1 , NUM_LANDMARKS );
        
        %%% Due to noise , DO NOT use all points for procrustes analysis
        %%% Divide face into 5 parts,randomly select 3 parts, randomly
        %%% select 5 points from each part
        if NUM_LANDMARKS == 68
            % randomly choose 3 parts from 5, for each part, choose 5
            % points randomly, repeat for 2 times, 20 times in total
            combines = nchoosek( 1:5 , 3 );
            % repeat for repeat times
            repeat = 1;
            all_shapes = zeros(NUM_LANDMARKS , 2 , size(combines,1)*repeat );
            % variation rank method to align the shape to mean shape space.
            for r = 1 : size(combines,1)
                % choose 3 parts
                first  = train_para.face_parts{ combines(r,1) };
                second = train_para.face_parts{ combines(r,2) };
                third  = train_para.face_parts{ combines(r,3) };
                
                for c = 1 : repeat
                    first_5  = datasample( first ,  5 , 'replace' , false );
                    second_5 = datasample( second , 5 , 'replace' , false );
                    third_5  = datasample( third ,  5 , 'replace' , false );
                    samples = [ first_5 second_5 third_5 ];
                    % Transform S_current to the mean shape space
                    % NOTE: Flipping is ALLOWED.
                    [d , z , gama] = procrustes( meanShape(:,samples)' , shape(:,samples)' , 'reflection' , false );
                    index = (r-1) * repeat + c ;
                    %gamas{ index } = gama;
                    all_shapes(:,:,index) = gama.b * shape' * gama.T + repmat( gama.c(1,:) , NUM_LANDMARKS , 1 );
                    de = 0;
                end
            end
        end    
        if NUM_LANDMARKS == 29
            % randomly choose 3 parts from 4, for each part, choose 4
            % points randomly, repeat for 3 times, 12 times in total
            combines = nchoosek( 1:4 , 3 );
            % repeat for 3 times
            repeat = 2;
            all_shapes = zeros(NUM_LANDMARKS , 2 , size(combines,1)*repeat );
            % variation rank method to align the shape to mean shape space.
            for r = 1 : size(combines,1)
                % choose 3 parts
                first  = train_para.face_parts{ combines(r,1) };
                second = train_para.face_parts{ combines(r,2) };
                third  = train_para.face_parts{ combines(r,3) };
                
                for c = 1 : repeat
                    first_4  = datasample( first ,  4 , 'replace' , false );
                    second_4 = datasample( second , 4 , 'replace' , false );
                    third_4  = datasample( third ,  4 , 'replace' , false );
                    samples = [ first_4 second_4 third_4 ];
                    % Transform S_current to the mean shape space
                    % NOTE: Flipping is ALLOWED.
                    [d , z , gama] = procrustes( meanShape(:,samples)' , shape(:,samples)' , 'reflection' , false );
                    index = (r-1)*repeat+c ;
                    %gamas{ index } = gama;
                    all_shapes(:,:,index) = gama.b * shape' * gama.T + repmat( gama.c(1,:) , NUM_LANDMARKS , 1 );
                    de = 0;
                end
            end
        end
        %%%%% Choose the stable points
        vars = var( all_shapes , 0 , 3 );
        vars = sum( vars.^2 , 2 );
        [tmp , index] = sort( vars );
        ind = index( 1 : train_para.var_rank_para );
            
        [d,z,gama] = procrustes( meanShape(:,ind)' , shape(:,ind)' , 'reflection' , false );
        aligned_shape = ( gama.b * shape' * gama.T + repmat( gama.c(1,:) , NUM_LANDMARKS , 1 ) )';
        
        % Centerize the shape
        if strcmp( train_para.center_shape , 'center' )
            center = mean( aligned_shape , 2 );
        elseif strcmp( train_para.center_shape , 'eyes_center' )
            if NUM_LANDMARKS == 29
                center = mean( aligned_shape(:,[9 11 13 14 12 10 15 16 17 18]) , 2 );
            elseif NUM_LANDMARKS == 68
                center = mean( aligned_shape(:,[37:42 43:48]) , 2 );
            else error('invalid option');end
        elseif strcmp( train_para.center_shape , 'nose_tip' )
            if NUM_LANDMARKS == 29
                center = aligned_shape(:,21);
            elseif NUM_LANDMARKS == 68
                center = aligned_shape(:,31);
            else error('invalid option');end
        elseif strcmp( train_para.center_shape , '1_28' )
            if( NUM_LANDMARKS ~= 29 )
                error('invalid');
            end
            center = mean( aligned_shape(:,1:28), 2 );
        else error('invalid option'); end
        
        aligned_shape = aligned_shape - repmat( center , 1 , NUM_LANDMARKS );
        gama.c(1,:) = gama.c(1,:) - center' ;
        gama.c = repmat( gama.c(1,:) , NUM_LANDMARKS , 1 );
        
        % Store the rs
        rs.gamas{i} = gama;
        rs.shape_centers(:,i) = center_shape;
        rs.phis(i,:) = [ aligned_shape(1,:) aligned_shape(2,:) ];
        de = 0;
    end
end

function rs = align_back( phis , gamas , shape_centers )
    global NUM_LANDMARKS;
    assert( size(phis,2) == 2*NUM_LANDMARKS );
    assert( size(phis,1) == length(gamas) );
    
    rs.phis = zeros( size(phis,1) , size(phis,2) );
    for i = 1 : size(phis,1)
        phi = phis(i,:);
        shape = [ phi(1:NUM_LANDMARKS); phi(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
        gama = gamas{i};
        trans_shape = ( ( ( shape' - gama.c ) / gama.T ) / gama.b )';
        trans_shape = trans_shape + repmat( shape_centers(:,i) , 1 , NUM_LANDMARKS );
        rs.phis(i,:) = [ trans_shape(1,:) trans_shape(2,:) ];
    end
end

function rs = transform_back( phis , centers , scalings )
    global NUM_LANDMARKS;
    assert( size(phis,2) == 2*NUM_LANDMARKS );
    assert( size(phis,1) == size(centers,2) & size(centers,1) == 2 );
    assert( size(scalings,1) == 1 & size(phis,1) == length(scalings) );
    rs = zeros(size(phis));
    
    for i = 1 : size(phis,1)
        phi = phis(i,:);
        %%%% Scaling %%%%%
        phi = phi * scalings(i);
        %%%% Translation %%%%%
        shape = [phi(1:NUM_LANDMARKS); phi(NUM_LANDMARKS+1: 2*NUM_LANDMARKS) ];
        shape = shape + repmat( centers(:,i) , 1 , NUM_LANDMARKS);
        rs(i,:) = [ shape(1,:) shape(2,:) ];
    end
end

%%%%%% Sparse shape constraint 
% phis   : every row is a phi, NOTE: Normalized !!!!!!!!!
% D      : shape dictionary
% params : parameters, must contain 'lambda' field
function rs = ssc_for_norm( phis , D , params )
    global NUM_LANDMARKS;
    assert( size(phis,2) == 2*NUM_LANDMARKS );
    assert( size(D,1) == 2*NUM_LANDMARKS );
    assert( isfield( params , 'lambda' ) );
    lambda = params.lambda;
    if isfield( params, 'weights' )
        weights = params.weights;
    end
    rs.update_phis = zeros( size(phis) );
    verbose = 1;
    
    for i = 1 : size(phis,1)
        phi = phis(i,:);
        phi = phi';
        if isfield( params, 'weights' )
            weight = weights(i,:);
        end
        
        %%%% Sparse shape update
        if isfield( params, 'weights' )
            B = glmnet( D , phi , 'gaussian' , struct('lambda',lambda,'weights',weight') );
        else
            B = glmnet( D , phi , 'gaussian' , struct('lambda',lambda) );
        end
        update_phi = D * B.beta + B.a0;
        rs.update_phis(i,:) = update_phi';
        rs.alpha = B.beta;
        rs.a0    = B.a0;
        
        %%% Output info
        if verbose
            if mod( i , round( size(phis,1)/4 ) ) == 0
                df = B.df;
                a0 = B.a0;
                dif = update_phi - phi ;
                mae_dif = mean( abs( dif ) );
                max_abs = max( abs( phi ) );
                fprintf('df:%d a0:%.4f mae_dif:%.4f max_abs:%.4f \n' , df , a0 , mae_dif , max_abs );
            end
        end
    end
    
    
end

%%%%%% Sparse shape constraint 
% phis   : every row is a phi
% D      : shape dictionary
% params : parameters, must contain 'lambda' field
% Output:
%    rs  : transformed phis
function rs = ssc( phis , D , params )
    global NUM_LANDMARKS;
    global shape_constraint_lambda;
    assert( size(phis,2) == 2*NUM_LANDMARKS );
    assert( size(D,1) == 2*NUM_LANDMARKS );
    assert( isfield( params , 'lambda' ) );
    lambda = params.lambda;
    if isfield( params, 'weights' )
        weights = params.weights;
    end
    rs.update_phis = zeros( size(phis) );
    verbose = 1;         
    
    %%%% Normalize phis if phis are raw image coords
    is_normalized = 0;
    if max(max(abs(phis))) > 100
        is_normalized = 1;
        rs = shapeGt('normalize_shape' , phis );
        phis        = rs.new_shapes;
        centers     = rs.centers;
        scalings    = rs.scalings;
    end
    % ssc
    if isfield( params, 'weights' )
        rs = shapeGt( 'ssc_for_norm' , phis , D , struct('lambda' , lambda, 'weights', weights ) );
    else
        rs = shapeGt( 'ssc_for_norm' , phis , D , struct('lambda' , lambda ) );
    end
    rs = rs.update_phis;
    % transform back
    if is_normalized
        rs = shapeGt( 'transform_back' , rs , centers , scalings );
    end
       
end

%%%%%% Calculate ced curve
% loss       : vector, note: percentage ( e.g., IOD based )
% max_limit  : max limit value
% cnt        : how many data points
% total      : total num of samples
% Return :
% rs : the same size as limits
function [rs,xs] = ced( loss , max_limit , cnt )
    % check
    assert( (size(loss,1) == 1 || size(loss,2) == 1) && mean(loss) < 1 );
    assert( isscalar(max_limit) && max_limit > 0 && max_limit < 1 );
    assert( isscalar(cnt) && cnt > 0 && cnt < 1000 ); 
    total = length(loss);
    assert( isscalar(total) && total > 0 && total < 1000000 ); 
    
    
    limits = linspace( 0 , max_limit , cnt );
    n = length( limits );
    for i = 1 : n
        limit = limits(i);
        num = sum( loss <= limit ) ;
        rs(i) = num / total ;
    end
    
    % check result
    assert( all(rs <= 1) );
    xs = limits;
end

function overlap = calcOverlap( box1, box2 )
    % check
    assert( size(box1,1)==1 && size(box1,2)==4 && size(box2,1)==1 && size(box2,2)==4 );
    assert( box1(3) >0 && box1(4) >0 && box2(3) >0 && box2(4) >0 );
    
    left1  = box1(1);
    up1    = box1(2);
    right1 = box1(1) + box1(3);
    down1  = box1(2) + box1(4);
    
    left2  = box2(1);
    up2    = box2(2);
    right2 = box2(1) + box2(3);
    down2  = box2(2) + box2(4);
    
    % calculate overlap area
    if right2 <= left1 || right1 <= left2 || down2 <= up1 || down1 <= up2
        overlap = 0;
        return;
    end
    
    inter = sort( [left1 right1 left2 right2] );
    inter_x = inter(3) - inter(2);
    assert( inter_x > 0 );
    inter = sort( [up1 down1 up2 down2] );
    inter_y = inter(3) - inter(2);
    assert( inter_y > 0 );
    
    overlap = inter_x * inter_y;

end

% Calculate overlap ratio with respect to the box1
function overlap_ratio = calcOverlapRatio( box1, box2 )
    % check
    assert( size(box1,1)==1 && size(box1,2)==4 && size(box2,1)==1 && size(box2,2)==4 );
    assert( box1(3) >0 && box1(4) >0 && box2(3) >0 && box2(4) >0 );
    
    left1  = box1(1);
    up1    = box1(2);
    right1 = box1(1) + box1(3);
    down1  = box1(2) + box1(4);
    
    left2  = box2(1);
    up2    = box2(2);
    right2 = box2(1) + box2(3);
    down2  = box2(2) + box2(4);
    
    % calculate overlap area
    if right2 <= left1 || right1 <= left2 || down2 <= up1 || down1 <= up2
        overlap_ratio = 0;
        return;
    end
    
    inter = sort( [left1 right1 left2 right2] );
    inter_x = inter(3) - inter(2);
    assert( inter_x > 0 );
    inter = sort( [up1 down1 up2 down2] );
    inter_y = inter(3) - inter(2);
    assert( inter_y > 0 );
    
    overlap_ratio = inter_x * inter_y / ( box1(3)*box1(4) );

end

% Read coords of N landmarks from pts_file
% Input:
% pts_file: the file which contains coords of N points
% N: the num of landmarks
% Output:
% coords: 2 * N matrix containing coords of N points
function coords = read_coords( pts_file, N )
    assert( N > 0 );
    f_pose = fopen( pts_file , 'r' );
    % ignore headers
    str1 = fgets( f_pose );
    str2 = fgets( f_pose );
    str3 = fgets( f_pose );
    % read the pose , a little tricky ~
    coordinates = zeros( N , 2 );
    coordinates = fscanf( f_pose , '%f' , N*2 );
    coords = reshape( coordinates , [ 2 , N ] );
    fclose( f_pose );
end

% Read face bounding box
% Input:
% box_file : the file containing face box
% Output:
% box: [ left, upper, width, height ]
function box = read_box( box_file)
    f_box = fopen( box_file , 'r' );
    box = fscanf( f_box , '%f' );
    assert( length(box) == 4 );
    if size(box,1) > 1
        box = box';
    end
    fclose( f_box );
end

% Task 1: Crop the face image I based on the shape S
% Task 2: Normalize it to 1-norm column vector,i.e.,  atom
% Input:
% I: the face image( RGB )
% S: the facial shape, 2 * NUM_LANDMARKS matrix
% PADDING: Extra PADDING space
% Output:
% rs: 3 fields
%     atom: the standard & normalized atom
%     range:[ left, right, up, down]
%     scale: Magnitude of the Scaling 
function rs = crop_and_normalize2_atom( I, S )
    global NUM_LANDMARKS;
    global train_para;
    % check
    assert( size(S,1)==2 & size(S,2)==NUM_LANDMARKS  );
    assert( size(I,3) == 3 ); % Only accept 3-channel image
    % Exception: Out of range of I
    x_min = min(S(1,:));
    x_max = max(S(1,:));
    y_min = min(S(2,:));
    y_max = max(S(2,:));
    PADDING = train_para.padding;
    left  = round( max( x_min - PADDING, 1 ) );
    right = round( min( x_max + PADDING, size(I,2)) );
    up    = round( max( y_min - PADDING, 1 ) );
    down  = round( min( y_max + PADDING, size(I,1)) );
    rs.range = [ left, right, up, down ];
    
    roi = I( up:down, left:right,:);
    % Resize the roi
    std_roi = imresize( roi, [train_para.std_h, train_para.std_w] );
    % Transform into a column vector: atom
    std_roi = double( reshape( std_roi, numel(std_roi), 1 ) );
    Scaling = norm( std_roi );
    rs.scale = Scaling;
    rs.atom = std_roi / Scaling;
    de = 0;
end

function rs = crop_and_normalize2_atom_debug( I, S )
    global NUM_LANDMARKS;
    global train_para;
    % check
    assert( size(S,1)==2 & size(S,2)==NUM_LANDMARKS  );
    assert( size(I,3) == 3 ); % Only accept 3-channel image
    % Exception: Out of range of I
    x_min = min(S(1,:));
    x_max = max(S(1,:));
    y_min = min(S(2,:));
    y_max = max(S(2,:));
    PADDING = train_para.padding;
    left = round( max( x_min, 1 ) );
    right = round( min( x_max, size(I,2)));
    up = round( max( y_min, 1));
    down = round( min( y_max, size(I,1)));
    rs.range = [ left, right, up, down ];
    
    roi = I( up:down, left:right,:);
    % Resize the roi
    std_roi = imresize( roi, [train_para.std_h, train_para.std_w] );
    % Transform into a column vector: atom
    std_roi = double( reshape( std_roi, numel(std_roi), 1 ) );
    Scaling = norm( std_roi );
    rs.scale = Scaling;
    rs.atom = std_roi / Scaling;
    de = 0;
end

function rs = crop_and_normalize2_atom_similarity( I, S )
    global NUM_LANDMARKS;
    global train_para;
    % check
    assert( size(S,1)==2 & size(S,2)==NUM_LANDMARKS  );
    assert( size(I,3) == 3 ); % Only accept 3-channel image
    % Exception: Out of range of I
    x_min = min(S(1,:));
    x_max = max(S(1,:));
    y_min = min(S(2,:));
    y_max = max(S(2,:));
    
    % two eyes
    left_eye  = [ mean( S(:,[9,11,13,17,14]), 2 ) ]';
    right_eye = [ mean( S(:,[12,10,15,18,16]), 2 ) ]';
    moving = [ left_eye; right_eye ];
    
    % Align the face region by Similarity Transformation
    std_left_eye  = [ train_para.std_w * 0.24, train_para.std_h * 0.26 ];
    std_right_eye = [ train_para.std_w * 0.76, train_para.std_h * 0.26 ];
    fixed = [ std_left_eye; std_right_eye ];
    
    std_roi = Img_similarity_transform_based_on_two_control_points( I, moving, fixed );
    % range
    rs.range = [ 1, train_para.std_w+1, 1, train_para.std_h+1 ];
    
%     % debugging
%     close all; figure(1);
%     subplot(1,3,1); imshow(I);   title('original');
%     subplot(1,3,2); imshow(std_roi); title('before imresize');
%     de = 0;
%     % Resize the roi
    std_roi = imresize( std_roi, [train_para.std_h, train_para.std_w] );
%     subplot(1,3,3); imshow(std_roi); title('after imresize');
    
    
    % Transform into a column vector: atom
    std_roi = double( reshape( std_roi, numel(std_roi), 1 ) );
    Scaling = norm( std_roi );
    rs.scale = Scaling;
    rs.atom = std_roi / Scaling;
    de = 0;
end

% Transform from atom to original face region
% Input:
% atom: the atom
% rs: see crop_and_normalize2_atom
% I_rec: original face region
function I_rec = atom2_original_roi( atom, rs )
    global NUM_LANDMARKS;
    global train_para;
    assert( size(atom,2) == 1 );
    % Scaling
    atom = uint8( atom * rs.scale );
    % Reshape
    roi = reshape( atom, [train_para.std_h, train_para.std_w, 3] );
    % Resize
    I_rec = imresize( roi, [ rs.range(4)-rs.range(3)+1, rs.range(2)-rs.range(1)+1 ] );
    
end

% Transform from atom to original face region
% Input:
% atom: the atom
% rs: see crop_and_normalize2_atom
% I_rec: original face region
function I_rec = atom2_original_roi2( atom, scale, range )
    global NUM_LANDMARKS;
    global train_para;
    assert( size(atom,2) == 1 );
    % Scaling
    atom = uint8( atom * scale );
    % Reshape
    roi = reshape( atom, [train_para.std_h, train_para.std_w, 3] );
    % Resize
    I_rec = imresize( roi, [ range(4)-range(3)+1, range(2)-range(1)+1 ] );
    
end

% I: the image
% moving: [ x1 y1 ;
%           x2 y2 ]
% fixed:  [ x1' y1';
%           x2' y2' ]
function y = Img_similarity_transform_based_on_two_control_points(I, moving, fixed )
    global train_para;
    assert( size(moving,1)==2 & size(moving,2)==2 );
    assert( size(fixed,1)==2 & size(fixed,2)==2 );
    % Three estimation targets:
    % 1. Scaling
    % 2. Rotation
    % 3. Translation
    
    % 1. Scaling
    dis1 = sqrt( (moving(1,1)-moving(2,1)).^2 + (moving(1,2)-moving(2,2)).^2 );
    dis2 = sqrt( (fixed(1,1) -fixed(2,1)).^2  + (fixed(1,2)-fixed(2,2)).^2 );
    scaling = dis2 / dis1;
    I1 = imresize( I, scaling );
    new_moving = moving * scaling;
    
%     % Rotation
%     % cos(theta) = v1*v2 / ( |v1|*|v2| )
%     v1 = [ moving(2,:) - moving(1,:) ]';
%     v2 = [ fixed(2,:) - fixed(1,:) ]';
%     % check
%     assert( v1(1) > 0 & v2(1) > 0 );
%     theta = acos( v1'*v2 / ( norm(v1)*norm(v2) ) );
%     assert( abs(theta) < pi/2 );
%     % perform the rotation
%     center = mean( moving );
%     I2 = imrotate(I,theta * 180 / pi,'bicubic')

    % Translation( essentially, cropping )
    left = round( max( new_moving(1,1) - fixed(1,1) + 1, 1 ) );
    up   = round( max( new_moving(1,2) - fixed(1,2) + 1, 1 ) );
    right= round( min( left + train_para.std_w-1, size(I1,2)) );
    down = round( min( up   + train_para.std_h-1, size(I1,1)) );
    
    y = I1( up:down, left:right, : );
    
    
end

% Calculate Projected Initial Shape . Ref: H. Zhu et al., Better initialization for regression-based face alignment, Computers & Graphics (2017)
% cur_s5: Currently estimated 5 key points( e.g., by MTCNN ), 2 * 5 matrix
% s_train: true shapes( 29 or 68 landmarks ) of training set, 2 *
%          NUM_LANDMARKS * N, N is # training samples
% params: optional
% params.m : m similar shapes
function PIS_shape = get_Projected_Initial_Shape( cur_s5, s_train, params )
    global NUM_LANDMARKS;
    % check params
    assert( size(cur_s5,1)==2 & size(cur_s5,2)==5 );
    assert( size(s_train,1)==2 & size(s_train,2)==NUM_LANDMARKS & size(s_train,3) > 1 );
    assert( isfield( params, 'm' ) );
    %% Step 1: Transform cur_s5 and s_train into normalized column vector
    v5 = cur_s5 - repmat( mean(cur_s5,2), 1, 5 );
    v5 = reshape( v5, numel(v5), 1 ); % current shape
    for i = 1 : size( s_train, 3 )
        ss = s_train(:,:,i);
        if NUM_LANDMARKS == 29
            s5 = ss(:,[17 18 21 23 24]);
        elseif NUM_LANDMARKS == 68
            error('remains to be developed');
        else
            error('invalid');
        end
        ss = ss - repmat( mean(ss,2), 1, NUM_LANDMARKS );
        s5 = s5 - repmat( mean(s5,2), 1, NUM_LANDMARKS );
        tr_s(:,i)  = reshape( ss, numel(ss), 1 ); % full landmarks
        tr_s5(:,i) = reshape( s5, numel(s5), 1 ); % five landmarks
    end
    %% Step 2 : Find m most similar shapes with Cosine Similarity Distance
    for i = 1 : size(s_train, 3)
        cos_sims(i) = ( v5' * tr_s5(:,i) ) / ( norm(v5)*norm(tr_s5(:,i)) );
    end
    [~,ind] = sort( cos_sims , 'descend' );
    S_prior = mean( tr_s(:,ind(1:params.m)), 2 );
    S_prior = reshape( S_prior, 2, NUM_LANDMARKS );
    
    %% Output: the projected initial shape
    % Apply the similarity transformation
    if NUM_LANDMARKS == 29
        S_prior5 = S_prior(:,[17 18 21 23 24]);
    elseif NUM_LANDMARKS == 68
        error('remains to be developed');
    else
        error('invalid');
    end
    [d,z,transform] = procrustes( cur_s5', S_prior5', 'reflection', false );
    % apply transform to S_prior
    
    PIS_shape = transform.b * S_prior' * transform.T + repmat( transform.c(1,:), NUM_LANDMARKS, 1 );
    
end

function parts = atom2parts( atom )
global train_para;
    assert( size(atom,1)==train_para.PATCH_W*train_para.PATCH_H*3*5 & size(atom,2) ==1 );
    atom = atom * 2e4;
    atom = uint8( atom);
    patch_size = train_para.PATCH_W*train_para.PATCH_H*3;
    part1 = atom(1:patch_size);
    part1 = reshape( part1,train_para.PATCH_H, train_para.PATCH_W, 3);
    part2 = atom(  patch_size+1:2*patch_size);
    part2 = reshape( part2,train_para.PATCH_H, train_para.PATCH_W, 3);
    part3 = atom(2*patch_size+1:3*patch_size);
    part3 = reshape( part3,train_para.PATCH_H, train_para.PATCH_W, 3);
    part4 = atom(3*patch_size+1:4*patch_size);
    part4 = reshape( part4,train_para.PATCH_H, train_para.PATCH_W, 3);
    part5 = atom(4*patch_size+1:5*patch_size);
    part5 = reshape( part5,train_para.PATCH_H, train_para.PATCH_W, 3);
    parts{1} = part1;
    parts{2} = part2;
    parts{3} = part3;
    parts{4} = part4;
    parts{5} = part5;
end

function [ vec_parts, info ] = extract_patches( img, coords )
global NUM_LANDMARKS;
global train_para;
% check
assert( size(img,3) == 3 ); % RGB Image only
assert( size(coords,1)==2 & size(coords,2) == NUM_LANDMARKS );

    % % % Left eye
    if NUM_LANDMARKS == 29
        pts = coords(:,[1 3 5 6 9 11 13 17 14]);
    else
        error('invalid');
    end
    left  = round( max( min(pts(1,:))-train_para.PADDING , 1 ) );
    right = round( min( max(pts(1,:))+train_para.PADDING , size(img,2)) );
    up    = round( max( min(pts(2,:))-train_para.PADDING , 1 ) );
    down  = round( min( max(pts(2,:))+train_para.PADDING , size(img,1)) );
    % Exception 1: Out of range
    if left >= right || up >= down; error('invalid');end
    roi   = img( up:down, left:right,:);
    std_roi1 = imresize( roi, [ train_para.PATCH_H, train_para.PATCH_W ] );

    % % % Right eye
    if NUM_LANDMARKS == 29
        pts = coords(:,[4 2 7 8 12 10 15 18 16]);
    else
        error('invalid');
    end
    left  = round( max( min(pts(1,:))-train_para.PADDING , 1 ) );
    right = round( min( max(pts(1,:))+train_para.PADDING , size(img,2)) );
    up    = round( max( min(pts(2,:))-train_para.PADDING , 1 ) );
    down  = round( min( max(pts(2,:))+train_para.PADDING , size(img,1)) );
    % Exception 1: Out of range
    if left >= right || up >= down; error('invalid');end
    roi   = img( up:down, left:right,:);
    std_roi2 = imresize( roi, [ train_para.PATCH_H, train_para.PATCH_W ] );

    % % % Nose
    if NUM_LANDMARKS == 29
        pts = coords(:,[19 20 21 22]);
    else
        error('invalid');
    end
    left  = round( max( min(pts(1,:))-train_para.PADDING , 1 ) );
    right = round( min( max(pts(1,:))+train_para.PADDING , size(img,2)) );
    up    = round( max( min(pts(2,:))-train_para.PADDING , 1 ) );
    down  = round( min( max(pts(2,:))+train_para.PADDING , size(img,1)) );
    % Exception 1: Out of range
    if left >= right || up >= down; error('invalid');end
    roi   = img( up:down, left:right,:);
    std_roi3 = imresize( roi, [ train_para.PATCH_H, train_para.PATCH_W ] );

    % % % Mouth
    if NUM_LANDMARKS == 29
        pts = coords(:,[23 24 25 26 27 28]);
    else
        error('invalid');
    end
    left  = round( max( min(pts(1,:))-train_para.PADDING , 1 ) );
    right = round( min( max(pts(1,:))+train_para.PADDING , size(img,2)) );
    up    = round( max( min(pts(2,:))-train_para.PADDING , 1 ) );
    down  = round( min( max(pts(2,:))+train_para.PADDING , size(img,1)) );
    % Exception 1: Out of range
    if left >= right || up >= down; error('invalid');end
    roi   = img( up:down, left:right,:);
    std_roi4 = imresize( roi, [ train_para.PATCH_H, train_para.PATCH_W ] );

    % % % Chin
    if NUM_LANDMARKS == 29
        pts = coords(:,[29]);
    else
        error('invalid');
    end
    left  = round( max( min(pts(1,:))-train_para.PADDING , 1 ) );
    right = round( min( max(pts(1,:))+train_para.PADDING , size(img,2)) );
    up    = round( max( min(pts(2,:))-train_para.PADDING , 1 ) );
    down  = round( min( max(pts(2,:))+train_para.PADDING , size(img,1)) );
    % Exception 1: Out of range
    if left >= right || up >= down; error('invalid');end
    roi   = img( up:down, left:right,:);
    std_roi5 = imresize( roi, [ train_para.PATCH_H, train_para.PATCH_W ] );
    
    % % % Build one whole representation vector
    vec_std_roi = [ reshape(std_roi1,numel(std_roi1),1);...
                    reshape(std_roi2,numel(std_roi2),1);...
                    reshape(std_roi3,numel(std_roi3),1);...
                    reshape(std_roi4,numel(std_roi4),1);...
                    reshape(std_roi5,numel(std_roi5),1) ];
    vec_std_roi = double(vec_std_roi);
    info.vec_norms = norm(vec_std_roi);
    vec_std_roi = vec_std_roi / norm(vec_std_roi);
    vec_parts = vec_std_roi;
end

% Given a face image and shape(N landmarks), get the face region
function [ vec_region, info ] = extract_face_region( img, coords, option )
    global train_para;
    global NUM_LANDMARKS;
    % check
    if train_para.channels == 3
        if size(img,3) == 1
            img = repmat(img,1,1,3);
        elseif size(img,3) == 3
        else
            error('invalid image');
        end
    elseif train_para.channels == 1
        if size(img,3) == 1
        elseif size(img,3) == 3
            img = rgb2gray(img);
        else
            error('invalid image');
        end
    end
    if ~( size(coords,1)==2 & (size(coords,2)==29||size(coords,2)==68) )
        de = 0;
        error('invalid');
    end
    assert( train_para.padding >= 0 );
    if nargin == 3
        points = option.points;
    else
        points = 1:NUM_LANDMARKS;
    end
    
    % If NUM_LANDMAKRS == 68
    if size(coords,2) == 68
        points = [ 18:68 ];
    end
    if size(coords,2) == 29
        points = [ 1:28 ];
    end
    
    % Crop the face region
    coords = coords(:,points);
    left  = round( max( min(coords(1,:))-train_para.padding , 1 ) );
    right = round( min( max(coords(1,:))+train_para.padding , size(img,2)) );
    up    = round( max( min(coords(2,:))-train_para.padding , 1 ) );
    down  = round( min( max(coords(2,:))+train_para.padding , size(img,1)) );
    info.region = [ left right up down ];
    region = img( up:down, left:right, : );
    % Resize & Reshape the face region
    region = imresize( region, [ train_para.std_h train_para.std_w ] );
    vec_region = reshape( region, numel(region), 1 );
    % Scale to unit norm
    vec_region = double(vec_region);
    info.scaling = norm(vec_region);
    vec_region = vec_region / norm(vec_region);
end

% Inverse process of the above function
% Given 
function region = atom2face_region( atom, info )
    global train_para;
    global NUM_LANDMARKS;
    % check
    if ~( size(atom,1)==train_para.std_h*train_para.std_w*train_para.channels & size(atom,2)==1 )
        warning('Invalid');
        error('invalid');
        de = 0;
    end
    
    atom = atom * info.scaling;
    atom = uint8( round(atom) );
    % reshape & resize
    region = reshape( atom, [ train_para.std_h, train_para.std_w, train_para.channels ] );
    region = imresize( region, [ info.region(4)-info.region(3)+1, info.region(2)-info.region(1)+1 ] );
end

% Detect data outliers ,i.e., extremely large elements
% Input: data, n*1 column vector of double or matrix
%        method, which kind of algorithm to use. By default,'median'
% Output: ind, n*1 logical vector, in which 1 indicates outlier
% Reference: refer to 'isoutlier' Matlab built-in function
function [ind,info] = is_outlier(data,method)
    % check
    % currently, only 1 channel is considered
    assert( size(data,3) ==1 );
    if nargin == 1 
        method = 'median';
    end
    data_size = size(data);
    
    data = reshape( data, numel(data), 1 );
    if strcmp( method, 'median' )
        c = -1/(sqrt(2)*erfcinv(3/2));
        scaled_MAD = c * median(abs(data-median(data)));
        th = 0.3 * scaled_MAD;
        
        info.th = median( data) + 3*th;
        ind = ( data >= info.th );
        ind = reshape( ind, data_size );
        
        % Find conncected 
        
    elseif strcmp( method, 'mean' )
        scaled_std = std( data );
        th = 3 * scaled_std;
        
        info.th = mean(data)+2*th;
        ind = (data) >= info.th;
        ind = reshape( ind, data_size );
    else
        error('invalid');
    end
end

% Step 1: Detect outliers
% Step 2: Detect connected components
% Step 3: Detect occlusions
function [ind,info] = is_occlusion(data, method)
    % Step 1 : Detect outliers
    if nargin == 2
        [ind,info] = is_outlier(data,method);
    else
        [ind,info] = is_outlier(data);
    end
    % Step 2 : Detect connected components
    L = bwlabel(ind,4);
    
    % Step 3: Detect occlusions
    % two priors:
    % 1. Connected to the edge of the image
    % 2. Occlusion area / Image area >= a threshold
    n_labels = max(max(L)); % num of different labels(i.e., connected components)
    for l = 1 : n_labels
        [row,col] = find( L == l );
        n_points = length( row );
        assert(n_points>0);
        if isempty( find( row == 1) ) & isempty( find( col == 1) ) & isempty( find( row == size(ind,1)) ) & isempty( find( col == size(ind,2)) )
            for i = 1 : n_points
                ind( row(i) , col(i) ) = 0;
            end
        end
    end
   
end

function de = plot_ori_rec_err( vec_ori, vec_rec )
    % error
    e = abs( vec_ori - vec_rec );
    parts_err = atom2parts( e );
    parts_err{1} = mean( parts_err{1}, 3);
    parts_err{2} = mean( parts_err{2}, 3);
    parts_err{3} = mean( parts_err{3}, 3);
    parts_err{4} = mean( parts_err{4}, 3);
    parts_err{5} = mean( parts_err{5}, 3);
    close all; 
    % the original
    ori_parts = atom2parts( vec_ori );
    subplot(3,5,1); imshow(ori_parts{1}); title('Original');
    subplot(3,5,2); imshow(ori_parts{2}); title('Original');
    subplot(3,5,3); imshow(ori_parts{3}); title('Original');
    subplot(3,5,4); imshow(ori_parts{4}); title('Original');
    subplot(3,5,5); imshow(ori_parts{5}); title('Original');
    % the recovered
    rec_parts = atom2parts( vec_rec );
    subplot(3,5,1+5); imshow(rec_parts{1}); title('Recovered');
    subplot(3,5,2+5); imshow(rec_parts{2}); title('Recovered');
    subplot(3,5,3+5); imshow(rec_parts{3}); title('Recovered');
    subplot(3,5,4+5); imshow(rec_parts{4}); title('Recovered');
    subplot(3,5,5+5); imshow(rec_parts{5}); title('Recovered');
    % error
%     parts_err{1} = uint8(abs(double(ori_parts{1}) - double(rec_parts{1})));
%     parts_err{2} = uint8(abs(double(ori_parts{2}) - double(rec_parts{2})));
%     parts_err{3} = uint8(abs(double(ori_parts{3}) - double(rec_parts{3})));
%     parts_err{4} = uint8(abs(double(ori_parts{4}) - double(rec_parts{4})));
%     parts_err{5} = uint8(abs(double(ori_parts{5}) - double(rec_parts{5})));
    subplot(3,5,1+2*5); imagesc(parts_err{1}); title('Error');
    subplot(3,5,2+2*5); imagesc(parts_err{2}); title('Error');
    subplot(3,5,3+2*5); imagesc(parts_err{3}); title('Error');
    subplot(3,5,4+2*5); imagesc(parts_err{4}); title('Error');
    subplot(3,5,5+2*5); imagesc(parts_err{5}); title('Error');
    
    de = 0;
end

% Given gt label and est label, compute corresponding precision/recall
% gt  : can be row/column vector or matrix, takes two values only 0 / 1
% est : can be row/column vector or matrix, takes real values between[0,1]
% option:
%        K: the number of points of the curve
function rs = compute_precision_recall_curve( gt, est, option )
    %check
    C_gt = unique( gt );
    C_est = unique( est );
    assert( length(C_gt)==2 & length(C_est)>=2 ); 
    if nargin == 3 & isfield( option, 'K' )
        K = option.K;
    else
        K = 500;
    end
    
    % For gt, there are a few configurations:
    % [ 0 1 ]
    % [ -1 1 ]
    gt( gt ~= 1 ) = 0;
    
    % tranform to column vector
    gt = reshape( gt, numel(gt), 1);
    est = reshape( est, numel(est), 1);
    
    % Determine threshold
    if nargin == 3 & isfield( option, 'RCPR' ) & option.RCPR == 1
        minv = min( est ); maxv = max( est );
        ths = linspace(minv,maxv,K);
    else
        ths = linspace(0,1,K);
    end
    rs.precision = zeros(K,1);
    rs.recall    = zeros(K,1);
    for i = 1 : length(ths)
        th = ths(i);
        
        est_bak = est;
        est_bak( est_bak >= th ) = 1;
        est_bak( est_bak < th )  = 0;
        pr = compute_precision_recall( gt, est_bak );
        rs.precision(i) = pr.precision;
        rs.recall(i) = pr.recall;
    end
    
end

% Given gt label and est label, compute corresponding precision/recall
% gt  : can be row/column vector or matrix, takes two values only 0 / 1
% est : can be row/column vector or matrix, takes two values only 0 / 1
function rs = compute_precision_recall( gt, est )
    %check
    C_gt = unique( gt );
    C_est = unique( est );
    assert( length(C_gt)==2 & length(C_est)<=2 );
    
    % tranform to column vector
    gt = reshape( gt, numel(gt), 1);
    est = reshape( est, numel(est), 1);
    
    % Precision / Recall
    true_positive = sum( gt & est );
    positive = sum( est );
    true     = sum( gt );
    rs.precision = true_positive / positive;
    rs.recall    = true_positive / true;
end

% Build dense graph from NUM_LANDMARKS landmarks
% Input: coord of NUM_LANDMARKS landmarks
% Output: dense graph
function rs = build_dense_graph( shape )
    global NUM_LANDMARKS;
    global Delaunay_triangles_frontal;
    n = NUM_LANDMARKS;
    % check
    assert( size(shape,1) == 1 & size(shape,2)==2*n );
    shape = [ shape(1:n); shape(n+1:2*n) ];
    assert( numel(Delaunay_triangles_frontal)>0 );
    
    % Step one: Perform delaunay decomposition
    if n == 29
        extra_points = zeros( 2, size(Delaunay_triangles_frontal,1) );
        for i = 1 : size(Delaunay_triangles_frontal,1)
            % Obtain the triangle
            A = shape( :, Delaunay_triangles_frontal(i,1) );
            B = shape( :, Delaunay_triangles_frontal(i,2) );
            C = shape( :, Delaunay_triangles_frontal(i,3) );
            extra_points(:,i) = mean( [A B C], 2 );
        end
    elseif n == 68
    else
        error('remains to be developed');
    end
    
    rs = [ shape extra_points ];
end