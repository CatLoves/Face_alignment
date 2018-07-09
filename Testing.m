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
global feat_para;
global Exper_ID;
global Dim_Feat ;
global Dim_Phi;
global imgs_train;
global cnt_out_of_range;    % count how many points are out ot range test the features are extracted.
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
cd( PREPROCESSING_DIR );
close all;

%% Prepare for training
% Set feature para
marker = 'Original';
feat_para.BlockSize = 5;
feat_para.CellSize  = 8;
T = 5;
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
train_para.occlu_estimation_method = 'spams';
feature_selection_lambda = 230;
shape_constraint_lambda = 0.0001;
occlusion_detection_lambda = 30;

% Load Normalization file
TRAIN_OR_TEST = 'TEST';
load( [ 'Original', '_', DATA_SET, '_', TRAIN_OR_TEST, '_normalization.mat' ] );
fprintf('%s has been loaded\n', [ marker, '_', DATA_SET, '_', TRAIN_OR_TEST, '_normalization.mat has been loaded.' ] );

% for K-fold validation
% Exper_ID = [ 'K_fold_no_sparse_occlu_detection_',Exper_ID];
% load( [Exper_ID,'_indice.mat'], 'Kfold_ind' );
% map_test{2} = 1:length(arr_imgs_attri); assert(length(map_test{2})>0);
% map_test{1}=[]; map_test{3}=[];
% map{2} = 1:length( find(Kfold_ind~=1) ); map{1}=[]; map{3}=[];

load( ['map_',DATA_SET,'.mat'], 'map');
load( ['map_test_',DATA_SET,'.mat'], 'map_test');
load( ['mean_shapes_',DATA_SET,'.mat'], 'mean_shapes' );
Exper_ID = [ 'Block5_no_occlu_two_scales',Exper_ID];

%%%%% Load the shape dictionary
load( ['Original', '_', DATA_SET, '_shape_dict.mat'], 'D_shape' );
% D_shape{1} = D_shape{2};
% D_shape{3} = D_shape{2};
% map_test{1} = [];
% map_test{2} = [];
% map_test{3} = [];
% for i = 1 : length(arr_imgs_attri)
%     yaws(i) = arr_imgs_attri(i).pose_angles(1);
%     if yaws(i) < -10
%         map_test{1} = [map_test{1},i];
%     elseif yaws(i) > 10
%         map_test{3} = [map_test{3},i];
%     else
%         map_test{2} = [map_test{2},i];
%     end
% end

%%%%% Load the shape dictionary
L = train_para.L_num_initial_shapes;
Use_Prepared_Feature = 0;
% To compute the regression matrix Wt or just load it from file
Use_Prepared_Wt = 0;
losses_train = zeros( T+1 , 1 );    % Load relevant data
Dim_Phi = 2*NUM_LANDMARKS;
debug_shape = 0;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Dual Sparse Constrained Cascade Regression  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for m = 1 : 3
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Facial shape initialization: L initial shapes selection  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t = 0;
    % For higher speed, set N = 200
    N  = length( map_test{ m } ); if N ==0; continue; end
    %N = 1500;
    N1 = N ;
    train_N_value = N;
    assert( N1 > 0 );
    % TrainSet Argumentation by factor L
    % Format of pCur: every row is a phi( shape )
    % We generate L initial shapes for each training image, and regard it
    % as L different images !!!!!!! THIS IS VERY IMPORTANT !!!!
    pCur       = zeros( N1 , Dim_Phi );
    pTrue      = zeros( N1 , Dim_Phi );     % store images coords( normalized image , rather than normalized wrt bbox )
    pInit      = pCur;                      % store phi_all when t = 0
    
    for i = 1 : length( map_test{m} )
        item = arr_imgs_attri( map_test{m}(i) );
        % True shape
        true_shape = item.true_coord_all_landmarks;
        row_true_shape = [ true_shape(1,:) true_shape(2,:) ];
        pTrue( (i), : ) = row_true_shape;
        % Initial shape
        pCur( (i), : ) = item.init_shapes.(train_para.shape_initialization)(1,:);
        nme_T0( i ) = shapeGt('dist', model, pCur( (i), : ), pTrue( (i), : ) );

        % debugging : Display init shape & true shape
%         close all;figure(i);
%         subplot(1,2,1);tmp=disp_landmarks(item.img,true_shape,struct('box',item.MTCNN_box));imshow(tmp);title('true shape');
%         subplot(1,2,2);tmp=disp_landmarks(item.img,pCur( (i),:) ,struct('box',item.MTCNN_box));imshow(tmp);title('init shape');
    end
  
    % debugging
    close all;
    th = 0.2;   % if > 40%, regarded as outliers
    for i = 1 : length( map_test{m}(i) )
        item = arr_imgs_attri( map_test{m}(i) );
        nme = shapeGt('dist', model, pCur( (i), : ), pTrue( (i), : ) );
        if nme > th
            close all; figure(1);
            [~,name,ext] = fileparts( item.image_fullname );
            true_coord = item.true_coord_all_landmarks;
            official_box = item.official_box;
            mtcnn_box = item.MTCNN_box;
            keypoints = item.coord_just_keypoints;
            init_coord = pCur(i,:);
            subplot(1,3,1);tmp=disp_landmarks(item.img,true_coord,struct('box',official_box));imshow(tmp);title('true');
            subplot(1,3,2);tmp=disp_landmarks(item.img,keypoints,struct('box',mtcnn_box));imshow(tmp);title('keypoints');
            subplot(1,3,3);tmp=disp_landmarks(item.img,init_coord);imshow(tmp);title('init shape');
            set(gcf,'name',name );
        end
    end
    
    % Calculate initial losses_test
    pInit = pCur;
    assert( ~isempty(feature_selection_lambda) );
    assert( ~isempty(shape_constraint_lambda) );
    %loss_name = strcat( 'losses_m' , num2str(m) , '_N' , num2str(N) , '_L' , num2str(L) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda), '_InitShape_' , train_para.shape_initialization ,  '_T0_Train.mat' );  
    init_losses = shapeGt( 'dist', model, pCur, pTrue );
    %save( loss_name , 'init_losses' );
    fprintf('test m = %d t = %d : Initial losses_train = %.2f %% \n' , m , t , mean(init_losses)*100 );
    
    %%%% Debugging settings %%%%%
    %debug_stage = 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  cascade regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for t = 1 : T

        feature_lambda = feature_selection_lambda;
        feat_file = strcat( DATA_SET , '_' , 'Feat_m' , num2str(m) , '_t' , num2str(t) , '_L' , num2str(L) , '_N' , num2str(N) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda),  '_InitShape_' , train_para.shape_initialization , '.mat' );
        % FOR DEBUGGING only
        if Use_Prepared_Feature == 1 && fopen( feat_file ) ~= -1
            % Load from mat file
            if t == 1
                load( feat_file , 'Feat' , 'pCur' , 'pTrue' , 'pTar' );
            else
                load( feat_file , 'Feat' );
            end
            loss_tmp = mean( shapeGt( 'dist',model, pCur , pTrue ) );
            fprintf( '%s loaded. Loss = %.2f %% \n' , feat_file , 100*loss_tmp );
        else    % %%% Compute feature
            start = tic;
            Feat = zeros( N1 , Dim_Feat );
            Occlu_test = zeros( N1, NUM_LANDMARKS );
            for i = 1 : length( map_test{m} )
                index = map_test{m}( i );          % the global index of current image
                item = arr_imgs_attri(index); % current item
                for k = 1 : 1
                    shape_row = pCur( (i) , : );
                    shape = [shape_row(1:NUM_LANDMARKS); shape_row(NUM_LANDMARKS+1:2*NUM_LANDMARKS)];
                    Feat( (i) , : ) = multi_scale_fhog( item.img , shape_row , struct('feature', 'multi_scale_fhog', 'scales',feat_para.Scales,'num_points' , NUM_LANDMARKS , 'pad_mode' , 'replicate') );
                    % debugging only
%                     close all; tmp = disp_landmarks(item.img, shape,struct('box',item.official_box)); imshow(tmp);
%                     de = 0;

                    % Query the occlusion status of current (img,shape)
                    Occlu_test( i,:) = item.occlu_label;
                end
                % indicate out of range error
                if mod( i , int32(N/2.01) ) == 0
                    fprintf( 'Out of range rate:%.2f %% \n' , 100*cnt_out_of_range / (N*L*NUM_LANDMARKS) );
                end
            end
            % Save the feature && Corresponding pCur
            if t == 1
                %save( feat_file , 'Feat' , 'pCur' , 'pTrue' , '-v7.3' );
            else
                %save( feat_file , 'Feat' , '-v7.3' );   % To save memory,
                %we do not save Feat
            end
            fprintf( '%s not saved! Time elapsed: %.4f s\n' , feat_file , toc(start) ); 
            % clear imgs_train to reserve memory
            clear 'imgs_train';
        end
        
        %%%%%%%%%%%%%%%%%%%%%% Get Sparse Logistic Regression Matrix
        %%%%%%%%%%%%%%%%%%%%%% W_occlu
%         W_occlu_file = strcat( Exper_ID,'_',DATA_SET , '_' , 'W_occlu_m' , num2str(m) , '_t' , num2str(t) , '_L' , num2str(L) , '_N' , num2str(length(map{m})) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda) ,  '_InitShape_' , train_para.shape_initialization, '_BlockSize_',num2str(feat_para.BlockSize) , '.mat' );
%         load( W_occlu_file );
%         fprintf('%s has been loaded\n', W_occlu_file );
%         Occlu_test_est = zeros( N1, NUM_LANDMARKS );
%         for i = 1 : NUM_LANDMARKS
%             rs = exp( Feat * W_occlu(i).B + W_occlu(i).FitInfo.Intercept );
%             Occlu_test_est(:,i) = 1- 1 ./ (1+rs);
%         end
        % weight Feat with gt occlu
%         Occlu_test_est = Occlu_test;
%         weight = kron( 1-Occlu_test_est, ones(1,Dim_Feat/NUM_LANDMARKS) );
%         Feat = Feat .* weight;
        %         if exist(W_occlu_file,'file')
%             load( W_occlu_file );
%             fprintf('%s has been loaded\n', W_occlu_file );
%             Occlu_test_est = zeros( N1, NUM_LANDMARKS );
%             for i = 1 : NUM_LANDMARKS
%                 rs = exp( Feat * W_occlu(i).B + W_occlu(i).FitInfo.Intercept );
%                 Occlu_test_est(:,i) = 1- 1 ./ (1+rs);
%             end
%             %Feat = Feat .* weight;
%             % %%%%% Calculate weight 
%             weight_points = (1-Occlu_test_est) ;
%             % Transform the weight to [ 0.5,1 ]
%             weight_points = ( 1 - 0.5 ) * weight_points + 0.5;
%             weight = repmat( weight_points, 1, 2);
%             
%             % %%%%% Impose weight on the features
%             weight_feat = kron(weight_points,ones(1,Dim_Feat/NUM_LANDMARKS) );
%             Feat = Feat .* weight_feat;   % Do not put it on features
%         end
        
        %%%%%%%%%%%%%%%%%%%%%% Get Sparse Linear Regression Matrix Wt %%%%%%%%%%%%%%%%%%%%
        % test [ W11 W12 W13 ... W1n ] 
        train_N = length(map{m});
        W_file = strcat( Exper_ID,'_',DATA_SET , '_' , 'W_m' , num2str(m) , '_t' , num2str(t) , '_L' , num2str(L) , '_N' , num2str(train_N) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda),'_BlockSize_',num2str(feat_para.BlockSize) , '.mat' );
        %%%%%%%%%%%%%%%%%%%%%% Get Sparse Linear Regression Matrix Wt
        load( W_file );
        
        %%%%%%%%%%%%%%%%%%%%%%%% Sparse Feature Selection %%%%%%%%%%%%%%%%%%%%%%
        delta_S = ( W * Feat' + repmat( beta_0 , 1 , N1) )';
        pCur = pCur + delta_S;
        
%         close all;
%         th = 0.2;   % if > 40%, regarded as outliers
%         for i = 1 : length( arr_imgs_attri )
%             item = arr_imgs_attri(i);
%             nme = shapeGt('dist', model, pCur( (i), : ), pTrue( (i), : ) );
%             if nme > th & t >= 2
%                 close all; figure(1);
%                 [~,name,ext] = fileparts( item.image_fullname );
%                 true_coord = item.true_coord_all_landmarks;
%                 official_box = item.official_box;
%                 mtcnn_box = item.MTCNN_box;
%                 keypoints = item.coord_just_keypoints;
%                 init_coord = pCur(i,:);
%                 subplot(1,3,1);tmp=disp_landmarks(item.img,true_coord,struct('box',official_box));imshow(tmp);title('true');
%                 subplot(1,3,2);tmp=disp_landmarks(item.img,keypoints,struct('box',mtcnn_box));imshow(tmp);title('keypoints');
%                 subplot(1,3,3);tmp=disp_landmarks(item.img,pCur(i,:));imshow(tmp);title('init shape');
%                 set(gcf,'name',name );
%             end
%         end
        
        clear 'delta_S';
        % debug to see change of losses_train
        loss_name = strcat( 'losses_m' , num2str(m) , '_t' , num2str(t) , '_N' , num2str(N) , '_L' , num2str(L) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda), '_BlockSize_',num2str(feat_para.BlockSize) , '_afterWt_Train.mat' );  
        [losses_Wt losses_all_SF_CR] = shapeGt( 'dist',model, pCur, pTrue );
        save('losses_SF_CR.mat', 'losses_all_SF_CR' );
        %save( loss_name , 'pCur' , 'pTrue' , 'losses_Wt' );
        %fprintf('%s be saved.\n' , loss_name );
        losses_train_after_wt( t ) = mean(losses_Wt);
        fprintf('test m = %d t = %d : After sparse feature selection, losses_test = %.2f %% \n' , m , t , mean(losses_Wt)*100 );
        de = 0;

        % cleaning data
        clear 'delta_S';
        
        %%%%%%%%%%%%%%%%%%%%%%%% Sparse Shape Constraint %%%%%%%%%%%%%%%%%%%%%%
        ssc_start = tic;
        if isempty( mean_shapes )
            load( strcat( num2str(NUM_LANDMARKS) , '_points_mean_shapes.mat' ) , 'mean_shapes' );
        end
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
            pCur_trans_mean(:,i) = estimated_shape_atom;
            pTrue_trans_mean(:,i) = estimated_shape_true;
            
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Sparse Shape Constraint %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            B = glmnet( D_shape{2} , estimated_shape_atom , 'gaussian', struct('lambda' , shape_constraint_lambda,'intr',0) );
            update_shape_atom = D_shape{2} * B.beta;

            %%%%%%%%%%%%%%%%%%%%% Transform back !  %%%%%%%%%%%%%%%%%%%%%
            update_phi = reshape( update_shape_atom , 2 , NUM_LANDMARKS )';
            updated_shape = ( update_phi - repmat( gama.c(1,:) , NUM_LANDMARKS , 1 ) ) / gama.b * inv(gama.T);
            % Interesting fact: center( shape ) =center( updated_shape );
            updated_shape = updated_shape';
            %updated_shape = updated_shape + repmat( center_shape , 1 , NUM_LANDMARKS );
            % Let nose tip be the origin
            %updated_shape = updated_shape + repmat( origin , 1 , NUM_LANDMARKS );
            % Convert to phi form
            update_phi = [ updated_shape(1,:) updated_shape(2,:) ];

            %%%%%%%%%%%%%%%%%%%%%%%% Update pCur %%%%%%%%%%%%%%%%%%%%%%%%
            pCur( i ,: ) = update_phi;

            if mod( i , int32(N1/10.01) ) == 0 
                tmp_loss = mean( shapeGt( 'dist',model, pCur(i,:) , pTrue(i,:) ) );
                fprintf( 'The %d th image: Before SSC, NME = %.2f %% \n', i,  100 *shape_nme_beforeSSC );
                fprintf( 'The %d th image: After  SSC, NME = %.2f %% \n' ,i , 100 * tmp_loss  );               
            end

        end

        
        % Validate the effect of sparse shape constraint
        loss_name = strcat( marker,'_losses_m' , num2str(m) , '_t' , num2str(t) , '_N' , num2str(N) , '_L' , num2str(L) , '_lam1_' , num2str(feature_selection_lambda) , '_lam2_' , num2str(shape_constraint_lambda),'_BlockSize_',num2str(feat_para.BlockSize) , '_afterDa_Train.mat' );  
        losses_Da = shapeGt( 'dist', model , pCur, pTrue );
        %save( loss_name , 'losses_Da'  );
        losses_train(t+1) = mean( losses_Da );
        fprintf('test m = %d t = %d : After sparse shape constraint , losses_train = %.2f %% \n' , m , t , losses_train(t+1)*100 );
        fprintf('shape constraint lambda = %.7f \n' , shape_lambda ); 
        ssc_time = toc( ssc_start );
        fprintf('Time for SSC:%.2f second %.2f second per shape\n\n' , ssc_time , ssc_time/N1 );
        
        %% Debugging: Visualization
%         K = 30;
%         % Set losses here
%         losses = losses_Da; 
%         [sorted_nme, ind] = sort(losses,'descend');
%         L_tmp = size(pCur,1)/length(map_test{m});
%         for i = 1 : K
%             index = map_test{m}( ceil( ind(i)/L_tmp ) );
%             item = arr_imgs_attri(index);
%             [~,name,~] = fileparts( item.image_fullname );
%             save_folder = [ 'Debug_images/',Exper_ID,'_m',num2str(m), '_t', num2str(t) ];
%             if exist( save_folder, 'file') == 0
%                 mkdir( save_folder );
%             end
%             close all; figure(i);
%             subplot(1,4,1);tmp=disp_landmarks(item.img,pTrue(ind(i),:));imshow(tmp);title('true');
%             imwrite(tmp, [save_folder,'/',name,'_true.png'] );
%             subplot(1,4,2);tmp=disp_landmarks(item.img,item.coord_just_keypoints );imshow(tmp);title('keypoints');
%             imwrite(tmp, [save_folder,'/',name,'_keypoints.png'] );
%             subplot(1,4,3);tmp=disp_landmarks(item.img,pInit(ind(i),:)); imshow(tmp);title('init shape');
%             imwrite(tmp, [save_folder,'/',name,'_initShape.png'] );
%             subplot(1,4,4);tmp=disp_landmarks(item.img,pCur(ind(i),:)); imshow(tmp);title('current shape');
%             nme = shapeGt('dist',model,pCur(ind(i),:),pTrue(ind(i),:) );
%             imwrite(tmp, [save_folder,'/',name,'_curShape.png'] );
%             set(gcf,'name',name);
%         end
%         de = 0;
    end

end


%%%%%%%%%%%%%%%% Clear data for more momery %%%%%%%%%%%%%%%%
if exist('imgs_train') && ~isempty( imgs_train )
    clear 'imgs_train' ;
end