%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Training of P_DSC_CR algorithm  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  For a training image, choose the NUM_INITIAL_SHAPES shapes
%%%%%%%%%%%%%  whose 3D pose( yaw , pitch , roll ) are most similar to its
%%%%%%%%%%%%%  3D pose.
%%%%%%%%%%%%%  Because of this kind of training method, Testing of P_DSC_CR
%%%%%%%%%%%%%  Follows the Similar Way, for a test image, choose the shape
%%%%%%%%%%%%%  of the image whose 3D pose is most similar to it.
%%%%%%%%%%%%%  Inspired by RCPR Code, Structs and Matrice are frequency
%% %%%%%%%%%%%  Global vars declaration 
global train_para;
% Load data && Set parameters
global NUM_LANDMARKS;
% the current num of stage, initially 0
global t;
global occlusion_detection_lambda;
global feat_para;
global Exper_ID;
global Dim_Feat ;
global Dim_Phi;
global cnt_out_of_range;    % count how many points are out ot range Train the features are extracted.
global PREPROCESSING_DIR;
global N;
global shape_constraint_lambda;
global losses_train;
global feature_selection_lambda;
global losses_train_after_wt;
global train_N_value;
global TRAIN_OR_TEST;
global DATA_SET;
global mean_shapes;
global shape_lambda;
global marker;
global Delaunay_triangles_frontal;
cd( PREPROCESSING_DIR );
close all;

%% Parameters go here
% Set feature para
marker = 'Original';
feat_para.BlockSize = 5;
feat_para.CellSize  = 8;
Dim_Phi = 2*NUM_LANDMARKS;
feat_para.Scales = 2;
Dim_Feat = ( feat_para.BlockSize*feat_para.BlockSize*2 ) * 31 * (NUM_LANDMARKS);

% Set initialization method
train_para.shape_initialization = 'multivariate_normal_sampling';
train_para.box_type = 'MTCNN';
train_para.L_num_initial_shapes = 10;
Exper_ID = [train_para.shape_initialization,'_L',num2str(train_para.L_num_initial_shapes)];
debug_shape = 0;

% Important parameters
% Set sparse problem solvers
train_para.lasso_method = 'spams';
train_para.occlu_estimation_method = 'glmnet';
feature_selection_lambda = 200;
shape_constraint_lambda = 0.0001;
occlusion_detection_lambda = 0.01;

% Load Normalization file
TRAIN_OR_TEST = 'TRAIN';
load( [marker, '_', DATA_SET, '_', TRAIN_OR_TEST, '_normalization.mat' ] );

% If we divide the training set into three pose( left/frontal/right )
load( ['map_',DATA_SET,'.mat'], 'map');
load( ['mean_shapes_',DATA_SET,'.mat'], 'mean_shapes' );
% K-fold Cross Validation where K = 5
N = length(arr_imgs_attri);
map{1} = []; map{2} = 1:N; map{3} = [];
Exper_ID = [ 'Block5_occlu',Exper_ID];

%%%%% Load the shape dictionary
load( [marker, '_', DATA_SET, '_shape_dict.mat'], 'D_shape' );

% N is #training images
% L is the num of initial shapes for each training image
% 1 : Just load Feat from mat file
% 0 : Compute Feat
% Some parameters
Use_Prepared_Feature = 0;
% To compute the regression matrix Wt or just load it from file
Use_Prepared_Wt = 0;
L = train_para.L_num_initial_shapes;
T = train_para.T_stages;
t = 0;
losses_train = zeros( T+1 , 1 );    % Load relevant data
%%%%% Create log file( to be compatible with Server
log_file = [PREPROCESSING_DIR,'/Log_',Exper_ID,'_',DATA_SET,'_','_L' , num2str(L), '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda), '_BlockSize_',num2str(feat_para.BlockSize),'.txt'];
fp_log = fopen( log_file, 'w' );
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Dual Sparse Constrained Cascade Regression  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for m = 1 : 3
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Facial shape initialization: L initial shapes selection  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t = 0;
    % For higher speed, set N = 200
    N  = length( map{ m } );
    %N = 1500;
    N1 = N * L;
    train_N_value = N;
    if N1 == 0
        continue;
    end
    % TrainSet Argumentation by factor L
    % Format of pCur: every row is a phi( shape )
    % We generate L initial shapes for each training image, and regard it
    % as L different images !!!!!!! THIS IS VERY IMPORTANT !!!!
    pCur       = zeros( N1 , Dim_Phi );
    pTrue      = zeros( N1 , Dim_Phi );     % store images coords( normalized image , rather than normalized wrt bbox )
    pInit      = pCur;                      % store phi_all when t = 0
    shape_nme_beforeSSC = zeros(T,N1);
    shape_nme_afterSSC  = zeros(T,N1);
    shape_nme_afterSSC_not_weighted = zeros(T,N1);
    
    for i = 1 : length( map{m} )
        item = arr_imgs_attri(map{m}(i));
        % True shape
        true_shape = item.true_coord_all_landmarks;
        row_true_shape = [ true_shape(1,:) true_shape(2,:) ];
        pTrue( (i-1)*L+1: i*L, : ) = repmat( row_true_shape, L, 1 );
        % Initial shape
        init_shapes = item.init_shapes.(train_para.shape_initialization);
        pCur( (i-1)*L+1: i*L, : ) = init_shapes(1:L,:);

        % debugging : Display init shape & true shape
%         close all;figure(i);
%         subplot(1,4,1);tmp=disp_landmarks(item.img,true_shape,struct('box',item.MTCNN_box));imshow(tmp);title('true shape');
%         subplot(1,4,2);tmp=disp_landmarks(item.img,pCur( (i-1)*L+1,:) ,struct('box',item.MTCNN_box));imshow(tmp);title('init shape');
%         subplot(1,4,3);tmp=disp_landmarks(item.img,pCur( (i-1)*L+5,:) ,struct('box',item.MTCNN_box));imshow(tmp);title('init shape');
%         subplot(1,4,4);tmp=disp_landmarks(item.img,pCur( (i-1)*L+10,:) ,struct('box',item.MTCNN_box));imshow(tmp);title('init shape');
    end
  
    % Calculate initial losses_train
    pInit = pCur;
    assert( ~isempty(feature_selection_lambda) ); assert( ~isempty(shape_constraint_lambda) );
    loss_name = strcat( 'losses_m' , num2str(m) , '_N' , num2str(N) , '_L' , num2str(L) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda), '_BlockSize_',num2str(feat_para.BlockSize), '_T0_Train.mat' );  
    init_losses = shapeGt( 'dist', model, pCur, pTrue );
    %save( loss_name , 'init_losses' );
    fprintf('Train m = %d t = %d : Initial losses_train = %.2f %% \n' , m , t , mean(init_losses)*100 );
    fprintf(fp_log, 'Train m = %d t = %d : Initial losses_train = %.2f %% \n' , m , t , mean(init_losses)*100 );
    
    %%%% Debugging settings %%%%%
    %debug_stage = 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  cascade regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for t = 1 : T
        
        %%%%%%%%%%%% Calculate delta_S , which is our target of regression %%%%%%%%%%
        tmp_losses = shapeGt( 'dist',model, pCur, pTrue );
        tmp_loss   = mean( shapeGt( 'dist',model, pCur, pTrue ) );
        tmp_var = std( tmp_losses );
        fprintf('Train m = %d t = %d : Initial losses_train = %.2f %% Variance = %.4f \n' , m , t , tmp_loss * 100 , tmp_var );
        fprintf(fp_log, 'Train m = %d t = %d : Initial losses_train = %.2f %% Variance = %.4f \n' , m , t , tmp_loss * 100 , tmp_var );
        pTar = pTrue - pCur;
        
        %% %%%%%%%%%%%%%%%%%%%%%% Extract Shape-indexed Feature %%%%%%%%%%%%%%%%%%%%%%
        if ~exist('arr_imgs_attri','var')
            load( [marker, '_', DATA_SET, '_', TRAIN_OR_TEST, '_normalization.mat' ] );
        end
        cnt_out_of_range = 0;
        feature_lambda = feature_selection_lambda;
        feat_file = strcat( Exper_ID,'_',DATA_SET , '_' , 'Feat_m' , num2str(m) , '_t' , num2str(t) , '_L' , num2str(L) , '_N' , num2str(N) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda), '_BlockSize_',num2str(feat_para.BlockSize), '.mat' );
        % FOR DEBUGGING only
        if t == 1
            Use_Prepared_Feature = 1;
        else
            Use_Prepared_Feature = 0;
        end
        if Use_Prepared_Feature == 1 && fopen( feat_file ) ~= -1
            % Load from mat file
            if t == 1
                load( feat_file );
            else
                load( feat_file , 'Feat' );
            end
            loss_tmp = mean( shapeGt( 'dist',model, pCur , pTrue ) );
            fprintf( '%s loaded. Loss = %.2f %% \n' , feat_file , 100*loss_tmp );
            fprintf( fp_log, '%s loaded. Loss = %.2f %% \n' , feat_file , 100*loss_tmp );
        else    % %%% Compute feature
            start = tic;
            Feat = zeros( N1 , Dim_Feat );
            Occlu = zeros( N1, NUM_LANDMARKS );
            for i = 1 : length(map{m})
                index = map{m}( i );          % the global index of current image
                item = arr_imgs_attri(index); % current item
                for k = 1 : L
                    shape_row = pCur( (i-1)*L + k , : );
                    shape = [ shape_row(1:NUM_LANDMARKS); shape_row(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
                    Feat( (i-1)*L + k , : ) = multi_scale_fhog( item.img , shape_row , struct('feature' , 'multi_scale_fhog', 'scales', feat_para.Scales 'num_points' , (NUM_LANDMARKS) , 'pad_mode' , 'replicate') );
                    % Query the occlusion status of current (img,shape)
                    for j = 1 : size(shape,2)
                        x = round(shape(1,j)); y = round(shape(2,j));
                        % check
                        if x < 1; x =1; end 
                        if x > size(item.img,2); x=size(item.img,2);end
                        if y < 1; y =1; end
                        if y > size(item.img,1); y=size(item.img,1);end
                        Occlu( (i-1)*L + k , j ) = item.mask( y, x );
                    end
                    
                    % debugging only
%                     close all; 
%                     %occlu_label = item.occlu_label;
%                     
%                     subplot(1,2,1);tmp = disp_landmarks(item.img, shape); imshow(tmp);
%                     subplot(1,2,2);imshow( item.mask); title('mask');
%                     de = 0;
                end
                % indicate out of range error
                if mod( i , int32(N/2.01) ) == 0
                    fprintf( 'Out of range rate:%.2f %% \n' , 100*cnt_out_of_range / (N*L*NUM_LANDMARKS) );
                    fprintf( fp_log, 'Out of range rate:%.2f %% \n' , 100*cnt_out_of_range / (N*L*NUM_LANDMARKS) );
                end
                            
            end
            % Save the feature && Corresponding pCur
            if t == 1
                %save( feat_file , 'Feat' , 'pCur' , 'pTrue' , 'pTar' ,'Occlu', '-v7.3' );
            else
                % save( feat_file , 'Feat' , 'pCur' , 'pTrue' , 'pTar' ,'Occlu', '-v7.3' );   % To save memory,
                % we do not save Feat
            end
            fprintf( '%s successfully saved! Time elapsed: %.4f s\n' , feat_file , toc(start) ); 
            fprintf( fp_log, '%s successfully saved! Time elapsed: %.4f s\n' , feat_file , toc(start) ); 
        end
        clear 'arr_imgs_attri';
        
        %% %%%%%%%%%%%%%%%%%%%% Occlusion detection via sparse logistic regression  %%%%%%%%%%%%%%%%%%%%
        W_occlu_file = strcat( Exper_ID,'_',DATA_SET , '_' , 'W_occlu_m' , num2str(m) , '_t' , num2str(t) , '_L' , num2str(L) , '_N' , num2str(N) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda) , '_BlockSize_',num2str(feat_para.BlockSize) , '.mat' );
        fprintf('Estimating occlusion probability of each landmark ...\n');
        fprintf('Method: %s \n', train_para.occlu_estimation_method );
        Occlu_train_est = zeros( size(Occlu) );
        W_occlu = zeros( size(Feat,2), size(Occlu,2) );
        start = tic;
        
        if strcmp( train_para.occlu_estimation_method, 'spams' )
            % with Intercept
            para = struct( 'lambda', occlusion_detection_lambda, ...
                           'loss', 'logistic',...
                           'regul', 'l1',...
                           'numThreads',1,...
                           'intercept', false);
            Occlu_bak = Occlu;
            Occlu_bak( Occlu_bak ~= 1 ) = -1;
            W_occlu = mexLasso( Occlu_bak , Feat, para );
            est_raw = Feat * W_occlu;
            Occlu_train_est = 1 - 1 ./ ( 1+exp(est_raw) );
        else
            for i = 1 : size(Occlu,2)
                start_occlu = tic;
                %[B,FitInfo] = lassoglm( Feat, Occlu(:,i),'binomial','lambda',occlusion_detection_lambda);
                fit = glmnet( Feat, Occlu(:,i), 'binomial', struct('lambda',occlusion_detection_lambda,'intr',0) );
                B = fit.beta;

                time = toc( start_occlu );
                fprintf('Sparse Logistic Regression for the %d-th landmark\n',i);
                fprintf('Time: %.2f s. %.2f s per image\n', time, time/size(Feat,1) );
                fprintf(fp_log,'Sparse Logistic Regression for the %d-th landmark\n',i);
                fprintf(fp_log,'Time: %.2f s. %.2f s per image\n', time, time/size(Feat,1) );

                Occlu_train_est(:,i) = 1- 1 ./ (1+exp( Feat * B ));
                % Save data
                W_occlu(:,i) = B;
                %W_occlu(i).FitInfo = FitInfo;
            end
        end
        % %%%%% Calculate weight
        weight_points = (1-Occlu_train_est) ;

        % %%%%% Impose weight on the features
        weight_feat = kron(weight_points,ones(1,Dim_Feat/size(Occlu,2)) );
        Feat = Feat .* weight_feat;   % Do not put it on features
        clear 'weight_feat';

        % %%%%% Save weight for the testing
        t_occlu = toc(start);
        save( W_occlu_file, 'W_occlu','Occlu_train_est' );
        fprintf('%s has been saved\n',W_occlu_file);
        fprintf('Time of occlusion estimation: %.1f second. %.3f s per sample\n',t_occlu,t_occlu/size(Feat,1) );
        fprintf( fp_log, '%s has been saved\n',W_occlu_file);
        % Report Precison/Recall 
        pr_train = shapeGt('compute_precision_recall_curve', Occlu, Occlu_train_est );
        indice = find( pr_train.precision >= 0.7 & pr_train.precision <= 0.9 ); assert( length(indice) > 0 );
        ind = indice( round(length(indice)/2) );
        fprintf('Precision/Recall: %.2f %% / %.2f %% \n', pr_train.precision(ind),pr_train.recall(ind) );
        fprintf( fp_log, 'Precision/Recall: %.2f %% / %.2f %% \n', pr_train.precision(ind),pr_train.recall(ind) );

     
        
        %% %%%%%%%%%%%%%%%%%%%% Get Sparse Linear Regression Matrix Wt %%%%%%%%%%%%%%%%%%%%
        % test [ W11 W12 W13 ... W1n ] 
        W_file = strcat( Exper_ID,'_',DATA_SET , '_' , 'W_m' , num2str(m) , '_t' , num2str(t) , '_L' , num2str(L) , '_N' , num2str(N) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda),'_BlockSize_',num2str(feat_para.BlockSize) , '.mat' );
        % for debugging only
        if t == 1
            Use_Prepared_Wt = 0;
        else
            Use_Prepared_Wt = 0;
        end
        %%%%%%%%%%%%%%%%%%%%%% Get Sparse Linear Regression Matrix Wt
        beta_0 = zeros( Dim_Phi , 1 );
        if Use_Prepared_Wt == 1 && fopen( W_file ) ~= -1
            % Load from mat file
            load( W_file );
            fprintf( '%s has been loaded.\n' , W_file );
            fprintf( fp_log, '%s has been loaded.\n' , W_file );
        else        % %%%% To compute the sparse linear regression matrix Wt
            W  = zeros( Dim_Phi , Dim_Feat ); beta_0 =zeros(Dim_Phi,1);
            if feature_selection_lambda == 0; error('invalid'); end
            fprintf('Lasso method: %s \n', train_para.lasso_method );
            fprintf('Lambda: %.4f \n', feature_selection_lambda );
            
            start = tic;
            if strcmp( train_para.lasso_method , 'spams' )
                para = struct('lambda',feature_selection_lambda,...
                              'numThreads', 1 );
                W = mexLasso( pTar, Feat, para );
                W = W';
                beta_0 = zeros( Dim_Phi, 1 );
            else
                for i = 1 : Dim_Phi
                    tar = pTar(:, i );
                    
                    if strcmp( train_para.lasso_method , 'matlab' )   
                        [ B_lasso , fitInfo ] = lasso( Feat , tar , 'lambda' , feature_selection_lambda );
                        W( i , : ) = B_lasso;
                        beta_0(i) = fitInfo.Intercept;
                    elseif strcmp( train_para.lasso_method , 'glmnet' )
                        B = glmnet( Feat , tar , 'gaussian' , struct( 'lambda' , feature_selection_lambda, 'intr', 0 ) );
                        W( i , : ) = B.beta;
                        beta_0( i ) = B.a0;
                    else
                        error('Invalid lasso method !');
                    end

                    % progress indicator      
                    if mod( i , int32(Dim_Phi/5.01) ) == 0
                        fprintf( 'lasso progress: %.2f %%\n' , 100* i / Dim_Phi );
                        fprintf( fp_log, 'lasso progress: %.2f %%\n' , 100* i / Dim_Phi );
                    end
                end
            end
            time = toc(start);
            fprintf('Computing Wt: %.4f second && %.2f s per image\n\n' , time , time / size(Feat,1) );
            fprintf(fp_log, 'Computing Wt: %.4f second && %.2f s per image\n\n' , time , time / size(Feat,1) );
            fprintf('Lasso method:%s\n' , train_para.lasso_method );
            fprintf(fp_log,'Lasso method:%s\n' , train_para.lasso_method );
            % disp parameters
            fprintf('Sparsity of Wt : %.2f %% \n' , 100*( 1 - nnz(W) / prod(size(W)) ) );
            fprintf(fp_log,'Sparsity of Wt : %.2f %% \n' , 100*( 1 - nnz(W) / prod(size(W)) ) );
            
            % save Wt
            save( W_file , 'W' , 'beta_0' );
            fprintf( 'Train Wt %s has been successfully saved.\n' , W_file );
            fprintf( fp_log, 'Train Wt %s has been successfully saved.\n' , W_file );
        end
        
%         %%%%%%%%%%%%%%%%%%%%%%%% Sparse Feature Selection %%%%%%%%%%%%%%%%%%%%%%
        delta_S = ( W * Feat' + repmat( beta_0 , 1 , N1) )';
        pCur = pCur + delta_S;
        clear 'W' 'beta_0';
        
        % debugging
%         close all;
%         th = 0.4;   % if > 40%, regarded as outliers
%         for i = 1 : size(pCur,1)
%             index = ceil( i / L );
%             item = arr_imgs_attri( index );
%             nme = shapeGt('dist', model, pCur( (i), : ), pTrue( (i), : ) );
%             if nme > th 
%                 close all; figure(1);
%                 [~,name,ext] = fileparts( item.image_fullname );
%                 true_coord = item.true_coord_all_landmarks;
%                 official_box = item.official_box;
%                 mtcnn_box = item.MTCNN_box;
%                 keypoints = item.coord_just_keypoints;
%                 init_coord = pInit(i,:);
%                 cur_coord = pCur(i,:);
%                 subplot(1,4,1);tmp=disp_landmarks(item.img,true_coord,struct('box',official_box));imshow(tmp);title('true');
%                 subplot(1,4,2);tmp=disp_landmarks(item.img,keypoints,struct('box',mtcnn_box));imshow(tmp);title('keypoints');
%                 subplot(1,4,3);tmp=disp_landmarks(item.img,pInit(i,:));imshow(tmp);title('init shape');
%                 subplot(1,4,4);tmp=disp_landmarks(item.img,pCur(i,:));imshow(tmp);title('init shape');
%                 set(gcf,'name',name );
%                 i = i + 10;
%             end
%         end
        
        clear 'delta_S';
        % debug to see change of losses_train
        loss_name = strcat( Exper_ID,'_','losses_m' , num2str(m) , '_t' , num2str(t) , '_N' , num2str(N) , '_L' , num2str(L) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda),  '_BlockSize_',num2str(feat_para.BlockSize) , '_afterWt_Train.mat' );  
        losses_Wt = shapeGt( 'dist',model, pCur, pTrue );
        %save( loss_name , 'pCur' , 'pTrue' , 'losses_Wt' );
        fprintf('%s be saved.\n' , loss_name );
        fprintf(fp_log,'%s be saved.\n' , loss_name );
        losses_train_after_wt( t ) = mean(losses_Wt);
        fprintf('Train m = %d t = %d : After sparse feature selection, losses_train = %.2f %% \n' , m , t , mean(losses_Wt)*100 );
        fprintf(fp_log,'Train m = %d t = %d : After sparse feature selection, losses_train = %.2f %% \n' , m , t , mean(losses_Wt)*100 );
        de = 0;

        % cleaning data
        clear 'Feat' 'delta_S';
        
        %%%%%%%%%%%%%%%%%%%%%%%% Sparse Shape Constraint %%%%%%%%%%%%%%%%%%%%%%
        ssc_start = tic;
        %weight_shape = kron( weight_points, ones(1,2) );
        % Calculate weights for shapes, transform from [0,1] to [0.5,1]
        %weight_shape = (1-0.5) * weight_shape + 0.5;
        atom_mean_shape = mean_shapes{m}; % same format as phi( x1 x2...xn y1 y2...yn )
        mean_shape = reshape(atom_mean_shape,2,NUM_LANDMARKS);
        
        for i = 1 : N1
            % get current shape
            p_shape = pCur( i ,: );
            true_phi = pTrue( i , : );
            shape_nme_beforeSSC = shapeGt( 'dist',model, p_shape, true_phi );
            true_shape = [ true_phi(1:NUM_LANDMARKS) ; true_phi(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
            % shape is the current shape !!!
            shape = [p_shape(1:NUM_LANDMARKS); p_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) ];
            
            % Estimate Procrustes Analysis Transformation Parameter: Gama
            if NUM_LANDMARKS == 68
                % randomly choose 3 parts from 5, for each part, choose 5
                % points randomly, repeat for 2 times, 20 times in total
                all_shapes(:,:,20) = zeros(NUM_LANDMARKS,2);
                %gamas(12) = [];
                combines = nchoosek( 1:5 , 3 );
                % variation rank method to align the shape to mean shape space.
                for r = 1 : size(combines,1)
                    % choose 3 parts
                    first  = train_para.face_parts{ combines(r,1) };
                    second = train_para.face_parts{ combines(r,2) };
                    third  = train_para.face_parts{ combines(r,3) };
                    % repeat for 2 times
                    for c = 1 : 2
                        first_5  = datasample( first ,  5 , 'replace' , false );
                        second_5 = datasample( second , 5 , 'replace' , false );
                        third_5  = datasample( third ,  5 , 'replace' , false );
                        samples = [ first_5 second_5 third_5 ];
                        % Transform S_current to the mean shape space
                        % NOTE: Flipping is ALLOWED.
                        [d , z , gama] = procrustes( mean_shape(:,samples)' , shape(:,samples)' );
                        index = (r-1)*2+c ;
                        gamas{ index } = gama;
                        all_shapes(:,:,index) = gama.b * shape' * gama.T + repmat( gama.c(1,:) , NUM_LANDMARKS , 1 );
                        de = 0;
                    end
                end
            end    
            if NUM_LANDMARKS == 29
                % randomly choose 3 parts from 4, for each part, choose 4
                % points randomly, repeat for 3 times, 12 times in total
                all_shapes(:,:,12) = zeros(NUM_LANDMARKS,2);
                %gamas(12) = [];
                combines = nchoosek( 1:4 , 3 );
                % variation rank method to align the shape to mean shape space.
                for r = 1 : size(combines,1)
                    % choose 3 parts
                    first  = train_para.face_parts{ combines(r,1) };
                    second = train_para.face_parts{ combines(r,2) };
                    third  = train_para.face_parts{ combines(r,3) };
                    % repeat for 3 times
                    for c = 1 : 3
                        first_4  = datasample( first ,  4 , 'replace' , false );
                        second_4 = datasample( second , 4 , 'replace' , false );
                        third_4  = datasample( third ,  4 , 'replace' , false );
                        samples = [ first_4 second_4 third_4 ];
                        % Transform S_current to the mean shape space
                        % NOTE: Flipping is ALLOWED.
                        [d , z , gama] = procrustes( mean_shape(:,samples)' , shape(:,samples)' );
                        index = (r-1)*3+c ;
                        gamas{ index } = gama;
                        all_shapes(:,:,index) = gama.b * shape' * gama.T + repmat( gama.c(1,:) , NUM_LANDMARKS , 1 );
                        de = 0;
                    end
                end
            end
            % Choose the stable points to estimate the final gama
            vars = var( all_shapes , 0 , 3 );
            vars = sum( vars.^2 , 2 );
            [tmp , index] = sort( vars );
            ind = index( 1 : train_para.var_rank_para );
            [d , z , gama] = procrustes( mean_shape(:,ind)' , shape(:,ind)' );
            estimated_shape = gama.b * shape' * gama.T + repmat( gama.c(1,:) , NUM_LANDMARKS , 1 );
            estimated_shape = estimated_shape'; 
            % For debugging
            estimated_shape_true = gama.b * true_shape' * gama.T + repmat( gama.c(1,:) , NUM_LANDMARKS , 1 );
            estimated_shape_true = estimated_shape_true';

            %%%%%%%%%%%%%%%%%%%%%%%% Update current shape Xt by linear combination of atoms in Dictionary %%%%%%%%%%%%%%%%%%%%%%%%
            %%% NOTE : estimated_shape may not be centered at (x_hat , y_hat),but atoms in Dictionary are centered at(x_hat ,y_hat)
            if strcmp( train_para.shape_center , 'center' )
                hat = mean( estimated_shape , 2 );
                % estimated_phi of format [x1 y1 x2 y2 x3 y3 ... xn yn ]'
                estimated_shape = estimated_shape - repmat( hat , 1 , NUM_LANDMARKS );
                gama.c(1,:) = gama.c(1,:) - hat';
                gama.c = repmat( gama.c(1,:) , NUM_LANDMARKS , 1 );
                % for the true shape
                %hat_true = mean( estimated_shape_true , 2 );
                estimated_shape_true = estimated_shape_true - repmat( hat , 1 , NUM_LANDMARKS); % Center of estimated_shape_true be (0,0)
            end
            estimated_shape_atom = reshape( estimated_shape , 2*NUM_LANDMARKS , 1 );
            estimated_shape_true = reshape( estimated_shape_true , 2*NUM_LANDMARKS , 1 );
            % debugging only
            
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Sparse Shape Constraint %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            B = glmnet( D_shape{2} , estimated_shape_atom , 'gaussian', struct('lambda' , shape_constraint_lambda, 'intr',0) );
            update_shape_atom = D_shape{2} * B.beta;

            %%%%%%%%%%%%%%%%%%%%% Transform back !  %%%%%%%%%%%%%%%%%%%%%
            update_phi = reshape( update_shape_atom , 2 , NUM_LANDMARKS )';
            updated_shape = ( update_phi - repmat( gama.c(1,:) , NUM_LANDMARKS , 1 ) ) / gama.b / (gama.T);
            % Interesting fact: center( shape ) =center( updated_shape );
            updated_shape = updated_shape';
            % Convert to phi form
            update_phi = [ updated_shape(1,:) updated_shape(2,:) ];

            %%%%%%%%%%%%%%%%%%%%%%%% Update pCur %%%%%%%%%%%%%%%%%%%%%%%%
            pCur( i ,: ) = update_phi;

            if mod( i , int32(N1/10.01) ) == 0 
                tmp_loss = mean( shapeGt( 'dist',model, pCur(i,:) , pTrue(i,:) ) );
                fprintf( 'The %d th image: Before SSC, NME = %.2f %% \n', i,  100 *shape_nme_beforeSSC );
                fprintf( fp_log, 'The %d th image: Before SSC, NME = %.2f %% \n', i,  100 *shape_nme_beforeSSC );
                fprintf( 'The %d th image: After  SSC, NME = %.2f DF = %d %% \n' ,i , 100 * tmp_loss , B.df );
                fprintf( fp_log, 'The %d th image: After  SSC, NME = %.2f %% \n' ,i , 100 * tmp_loss  ); 
            end

        end

        
        % Validate the effect of sparse shape constraint
        loss_name = strcat( Exper_ID,'_',marker,'_losses_m' , num2str(m) , '_t' , num2str(t) , '_N' , num2str(N) , '_L' , num2str(L) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda), '_BlockSize_',num2str(feat_para.BlockSize) , '_afterDa_Train.mat' );  
        losses_Da = shapeGt( 'dist', model , pCur, pTrue );
        %save( loss_name , 'losses_Da'  );
        losses_train(t+1) = mean( losses_Da );
        fprintf('Train m = %d t = %d : After sparse shape constraint , losses_train = %.2f %% \n' , m , t , losses_train(t+1)*100 );
        fprintf(fp_log,'Train m = %d t = %d : After sparse shape constraint , losses_train = %.2f %% \n' , m , t , losses_train(t+1)*100 );
        fprintf('shape constraint lambda = %.7f \n' , shape_lambda );
        fprintf(fp_log, 'shape constraint lambda = %.7f \n' , shape_lambda );
        
        ssc_time = toc( ssc_start );
        fprintf('Time for SSC:%.2f second %.2f second per shape\n\n' , ssc_time , ssc_time/N1 );
        fprintf(fp_log, 'Time for SSC:%.2f second %.2f second per shape\n\n' , ssc_time , ssc_time/N1 );
        
        de = 0;
    end

end


%%%%%%%%%%%%%%%% Clear data for more momery %%%%%%%%%%%%%%%%
fclose( fp_log );