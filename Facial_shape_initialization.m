%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Facial shape initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Input: Normalization_file
% % Output: Assign L initial shapes to each training sample
% %         Assign 1 initial shape to each testing sample
% %         Field name:init_shapes
% % Initialization of facial shapes
% % There are various initialization methods, new methods may come out soon
% % Typical methods:
% % 1. Simple comparing: Similarity based on some kind of features(LBP,LDP,HOG,ect)
% % 2. Image retrieval: Bag of words, etc
% % 3. Dictionary learning: relational dictionary learning
% % ...
%% Environmental variables
global TRAIN_OR_TEST;
global marker;
global train_para;
global NUM_LANDMARKS;
global model;
global DATA_SET;
global Exper_ID;

%% Which initialization method should be used ?
%train_para.shape_initialization = 'mean_shape_with_Gaussian_sampling';
% Possible initialization methods:
% official_box_mean_shape_with_Gaussian_noise_level_0_05_with_3pose
% mtcnn_box_mean_shape_with_Gaussian_noise_level_0_05_with_3pose
% from_5_to_all_with_pose_similarity_Euclidean
% modified_offcial_box_mean_shape_with_Gaussian_noise_level_0_05
% official_box_mean_shape_with_Gaussian_noise_level_mean_Std
% MTCNN_box_mean_shape_with_Gaussian_noise_level_mean_Std
% MTCNN_box_mean_shape_with_Multivariate_Normal_Sampling
% official_box_mean_shape_with_Multivariate_Normal_Sampling
% simple_from_5_to_all
% comparing_with_LBP_official_box
% comparing_with_LBP_MTCNN_box
% LBP_simi_Pearson_crop_MTCNN_keypoints
% HOG_of_MTCNN_keypoints_ls
% mean_shape_with_box_augmentation_MTCNN
% mean_shape_with_box_augmentation_official
% mean_shape_with_box_aug_crop_MTCNN_keypoints
% mean_shape_with_box_aug_crop_MTCNN_keypoints_poses
% mean_shape_with_box_aug_crop_true_keypoints_poses
% RCPR_box_aug_rand_shapes
% MTCNN_box_rand_shapes
% multivariate_normal_sampling

train_para.shape_initialization = 'multivariate_normal_sampling';
marker = 'Original';
train_para.box_type = 'crop_MTCNN_keypoints';
train_para.L_num_initial_shapes = 20;
L = train_para.L_num_initial_shapes;
Exper_ID = [train_para.shape_initialization,'_L',num2str(train_para.L_num_initial_shapes)];
fprintf('********** Initialization method: %s\n', train_para.shape_initialization ); 
fprintf('********** BoundingBox type: %s\n', train_para.box_type ); 

if strcmp( train_para.shape_initialization , 'official_box_mean_shape_with_Gaussian_noise_level_0_05_with_3pose' )
    if ~exist('arr_imgs_attri_train' )
        Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
        load( Normalization_file );
        arr_imgs_attri_train = arr_imgs_attri;
    end
    if ~exist('arr_imgs_attri_test' )
        Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
        load( Normalization_file );
        arr_imgs_attri_test = arr_imgs_attri;
    end
    N = length(arr_imgs_attri_train);
    L = train_para.L_num_initial_shapes;
    
    
    
end

if strcmp( train_para.shape_initialization , 'official_box_mean_shape_with_Gaussian_noise_level_0_05_with_3pose' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
	arr_imgs_attri_train = arr_imgs_attri;
	Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
	arr_imgs_attri_test = arr_imgs_attri;
    N = length(arr_imgs_attri_train);
    L = train_para.L_num_initial_shapes;
    % Divide the training & test set into 3 poses
    load( ['map_',DATA_SET,'.mat'], 'map');
    load( ['map_test_',DATA_SET,'.mat'], 'map_test');
    
    for m = 1 : 3
        % Get shapes w.r.t MTCNN box of all training samples
        shapes_wrt_box = zeros( length(map{m}), 2*NUM_LANDMARKS );
        N = length(map{m});
        for i = 1 : length( map{m} )
            item = arr_imgs_attri_train( map{m}(i) );
            true_coords = item.true_coord_all_landmarks;
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            else
                error('remain to be developed.');
            end
            norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
            norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
            shapes_wrt_box(i,:) = norm_shape;
        end

        % Build the multivariate normal distribution model for
        % shapes_wrt_box
        mean_shapes_wrt_box = mean( shapes_wrt_box );
        std_shapes_wrt_box  = mean( std(shapes_wrt_box) );
        noise_level = 0.5 * std_shapes_wrt_box;
        samples = repmat( mean_shapes_wrt_box, N*L, 1 ) + random('norm',0,noise_level,N*L,2*NUM_LANDMARKS);
        % Generate initial shapes for each training sample( raw image
        % coords )
        pTrue = zeros( N*L, 2*NUM_LANDMARKS);
        pInit = zeros( N*L, 2*NUM_LANDMARKS);
        for i = 1 : N
            item = arr_imgs_attri_train( map{m}(i) );
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            else
                error('remain to be developed.');
            end
            true_shape = item.true_coord_all_landmarks;
            true_shape_row = [true_shape(1,:) true_shape(2,:)];
            pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, train_para.L_num_initial_shapes,1);
            init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:), repmat( box, train_para.L_num_initial_shapes,1) );
            arr_imgs_attri_train( map{m}(i) ).init_shapes.(train_para.shape_initialization) = init_shapes;
            pInit( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = init_shapes;
    %             % debugging : display
    %             close all; figure(i);
    %             subplot(1,4,1); tmp=disp_landmarks(arr_imgs_attri_train(i).img, arr_imgs_attri_train(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri_train(i).MTCNN_box));imshow(tmp);
    %             subplot(1,4,2); tmp=disp_landmarks(arr_imgs_attri_train(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri_train(i).MTCNN_box));imshow(tmp);
    %             subplot(1,4,3); tmp=disp_landmarks(arr_imgs_attri_train(i).img, init_shapes(3,:),  struct('box',arr_imgs_attri_train(i).MTCNN_box));imshow(tmp);
    %             subplot(1,4,4); tmp=disp_landmarks(arr_imgs_attri_train(i).img, init_shapes(5,:),  struct('box',arr_imgs_attri_train(i).MTCNN_box));imshow(tmp);
    %             de = 0;
        end
        % Save the result
        nme_train(m) = mean( shapeGt( 'dist',model, pInit, pTrue ) );
        

        % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
		N_test = length( map_test{m} ); assert(N_test>0);
        pTrue_test = zeros( N_test, 2*NUM_LANDMARKS);
        pInit_test = zeros( N_test, 2*NUM_LANDMARKS);
        for i = 1 : N_test
            item = arr_imgs_attri_test( map_test{m}(i) );
            true_shape = item.true_coord_all_landmarks;
            true_shape_row = [true_shape(1,:) true_shape(2,:)];
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            else
                error('remain to be developed.');
            end

            pTrue_test(i,:) = true_shape_row;
            init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
            arr_imgs_attri_test(map_test{m}(i)).init_shapes.(train_para.shape_initialization) = init_shapes;
            pInit_test(i,:) = init_shapes;
    %         % debugging : display
    %         close all; figure(i);
    %         subplot(1,2,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,  struct('box',item.MTCNN_box));imshow(tmp);title('true');
    %         subplot(1,2,2); tmp=disp_landmarks(item.img, init_shapes(1,:),  struct('box',item.MTCNN_box));imshow(tmp); title('initial shape');
    % 
    %         de = 0;
        end
    end % for m = 1 : 3
    
    % Save the result
	arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
        
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
	arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );

end

if strcmp( train_para.shape_initialization , 'mtcnn_box_mean_shape_with_Gaussian_noise_level_0_05_with_3pose' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
	arr_imgs_attri_train = arr_imgs_attri;
	Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
	arr_imgs_attri_test = arr_imgs_attri;
    N = length(arr_imgs_attri_train);
    L = train_para.L_num_initial_shapes;
    % Divide the training & test set into 3 poses
    load( ['map_',DATA_SET,'.mat'], 'map');
    load( ['map_test_',DATA_SET,'.mat'], 'map_test');
    
    for m = 1 : 3
        % Get shapes w.r.t MTCNN box of all training samples
        shapes_wrt_box = zeros( length(map{m}), 2*NUM_LANDMARKS );
        N = length(map{m});
        for i = 1 : length( map{m} )
            item = arr_imgs_attri_train( map{m}(i) );
            true_coords = item.true_coord_all_landmarks;
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            else
                error('remain to be developed.');
            end
            norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
            norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
            shapes_wrt_box(i,:) = norm_shape;
        end

        % Build the multivariate normal distribution model for
        % shapes_wrt_box
        mean_shapes_wrt_box = mean( shapes_wrt_box );
        std_shapes_wrt_box  = mean( std(shapes_wrt_box) );
        noise_level = 0.5 * std_shapes_wrt_box;
        samples = repmat( mean_shapes_wrt_box, N*L, 1 ) + random('norm',0,noise_level,N*L,2*NUM_LANDMARKS);
        % Generate initial shapes for each training sample( raw image
        % coords )
        pTrue = zeros( N*L, 2*NUM_LANDMARKS);
        pInit = zeros( N*L, 2*NUM_LANDMARKS);
        for i = 1 : N
            item = arr_imgs_attri_train( map{m}(i) );
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            else
                error('remain to be developed.');
            end
            true_shape = item.true_coord_all_landmarks;
            true_shape_row = [true_shape(1,:) true_shape(2,:)];
            pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, train_para.L_num_initial_shapes,1);
            init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:), repmat( box, train_para.L_num_initial_shapes,1) );
            arr_imgs_attri_train( map{m}(i) ).init_shapes.(train_para.shape_initialization) = init_shapes;
            pInit( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = init_shapes;
    %             % debugging : display
    %             close all; figure(i);
    %             subplot(1,4,1); tmp=disp_landmarks(arr_imgs_attri_train(i).img, arr_imgs_attri_train(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri_train(i).MTCNN_box));imshow(tmp);
    %             subplot(1,4,2); tmp=disp_landmarks(arr_imgs_attri_train(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri_train(i).MTCNN_box));imshow(tmp);
    %             subplot(1,4,3); tmp=disp_landmarks(arr_imgs_attri_train(i).img, init_shapes(3,:),  struct('box',arr_imgs_attri_train(i).MTCNN_box));imshow(tmp);
    %             subplot(1,4,4); tmp=disp_landmarks(arr_imgs_attri_train(i).img, init_shapes(5,:),  struct('box',arr_imgs_attri_train(i).MTCNN_box));imshow(tmp);
    %             de = 0;
        end
        % Save the result
        nme_train(m) = mean( shapeGt( 'dist',model, pInit, pTrue ) );
        

        % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
		N_test = length( map_test{m} ); assert(N_test>0);
        pTrue_test = zeros( N_test, 2*NUM_LANDMARKS);
        pInit_test = zeros( N_test, 2*NUM_LANDMARKS);
        for i = 1 : N_test
            item = arr_imgs_attri_test( map_test{m}(i) );
            true_shape = item.true_coord_all_landmarks;
            true_shape_row = [true_shape(1,:) true_shape(2,:)];
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            else
                error('remain to be developed.');
            end

            pTrue_test(i,:) = true_shape_row;
            init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
            arr_imgs_attri_test(map_test{m}(i)).init_shapes.(train_para.shape_initialization) = init_shapes;
            pInit_test(i,:) = init_shapes;
    %         % debugging : display
    %         close all; figure(i);
    %         subplot(1,2,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,  struct('box',item.MTCNN_box));imshow(tmp);title('true');
    %         subplot(1,2,2); tmp=disp_landmarks(item.img, init_shapes(1,:),  struct('box',item.MTCNN_box));imshow(tmp); title('initial shape');
    % 
    %         de = 0;
        end
    end % for m = 1 : 3
    
    % Save the result
	arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
        
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
	arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );

end

if strcmp( train_para.shape_initialization , 'from_5_to_all_with_pose_similarity_Euclidean' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    if ~exist('arr_imgs_attri_train')
        load( Normalization_file );
		arr_imgs_attri_train = arr_imgs_attri;
    end
	Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    if ~exist('arr_imgs_attri_test')
        load( Normalization_file );
		arr_imgs_attri_test = arr_imgs_attri;
    end
	
	% Divide into 3 poses(left,frontal,right)
	load( ['map_',DATA_SET,'.mat'], 'map');
	load( ['map_test_',DATA_SET,'.mat'], 'map_test');
    map{2} = 1 : length(arr_imgs_attri_train);
    map{1}=[]; map{3}=[];
    map_test{2} = 1 : length(arr_imgs_attri_test);
    map_test{1}=[]; map_test{3}=[];
    save( ['map_',DATA_SET,'.mat'], 'map');
	save( ['map_test_',DATA_SET,'.mat'], 'map_test');
	for m = 1 : 3
		n = length( map{m} ); if n ==0; continue; end
		Tr_s5 = zeros( 10, length(arr_imgs_attri_train) );
		Tr_all = zeros( NUM_LANDMARKS*2, length(arr_imgs_attri_train) );
		lambda = 0.01;

		% Construct Tr_s5 and Tr_all
		for i = 1 : length( map{m} )
			item = arr_imgs_attri_train( map{m}(i) );
			true_coords = item.true_coord_all_landmarks;
			five_points = item.coord_just_keypoints;
			center = mean( five_points, 2 );
			norm_five_points = five_points - repmat( center, 1, 5 );
			norm_five_points = reshape(norm_five_points,numel(norm_five_points),1);
			norm_five_points = norm_five_points / norm(norm_five_points);
			Tr_s5(:,i) = norm_five_points;
			norm_true_coords = true_coords - repmat( center, 1, NUM_LANDMARKS );
			norm_true_coords = reshape(norm_true_coords,numel(norm_true_coords),1);
			Tr_all(:,i) = norm_true_coords;
		end

		% Compute initial shapes for training samples
		for i = 1 : length( map{m} )
			item = arr_imgs_attri_train( map{m}(i) );
			Tr_s5_bak = Tr_s5;
			Tr_all_bak = Tr_all;
			% Remove the training item itself
			Tr_s5_bak(:,i) = [];
			Tr_all_bak(:,i) = [];
			% Normalize five keypoints
			true_coords = item.true_coord_all_landmarks;
			five_points = item.coord_just_keypoints;
			center = mean( five_points, 2 );
			norm_five_points = five_points - repmat( center, 1, 5 );
			norm_five_points = reshape(norm_five_points,numel(norm_five_points),1);
			norm_five_points = norm_five_points / norm(norm_five_points);
			
			% Which kind of similarity measure to use ? Does it make much difference ?
			similarity_measure = 'Euclidean';
			dis = zeros( length(Tr_s5_bak),1 );
			init_shapes = zeros(L,NUM_LANDMARKS*2);
			
            for j = 1 : size(Tr_s5_bak,2)
                if strcmp( similarity_measure, 'Euclidean' )
                    dis(j) = norm( norm_five_points - Tr_s5_bak(:,j) );
                elseif strcmp( similarity_measure, 'Euclidean' )    
                else
                    error('remains to be developed');
                end
            end
            % sort
            [sorted_dis,ind] = sort( dis );
            selected = ind(1:L);
            for j = 1 : L
                index = selected(j);
                s = Tr_all_bak(:,index);
                shape = reshape(s,2,NUM_LANDMARKS);
                shape = shape + repmat( center, 1, NUM_LANDMARKS);
                init_shapes(j,:) = [ shape(1,:) shape(2,:) ];
            end
            arr_imgs_attri_train( map{m}(i) ).init_shapes.(train_para.shape_initialization) = init_shapes;

		end
		
		% Compute initial shapes for test samples
		for i = 1 : length( map_test{m} )
			item = arr_imgs_attri_test( map_test{m}(i) );
			% Normalize five keypoints
			true_coords = item.true_coord_all_landmarks;
			five_points = item.coord_just_keypoints;
			center = mean( five_points, 2 );
			norm_five_points = five_points - repmat( center, 1, 5 );
			norm_five_points = reshape(norm_five_points,numel(norm_five_points),1);
			norm_five_points = norm_five_points / norm(norm_five_points);
			
			% Which kind of similarity measure to use ? Does it make much difference ?
			dis = zeros( length(Tr_s5),1 );
			init_shapes = zeros(L,NUM_LANDMARKS*2);

            for j = 1 : size(Tr_s5,2)
                if strcmp( similarity_measure, 'Euclidean' )
                    dis(j) = norm( norm_five_points - Tr_s5(:,j) );
                elseif strcmp( similarity_measure, 'Euclidean' )    
                else
                    error('remains to be developed');
                end
            end
            % sort
            [sorted_dis,ind] = sort( dis );
            selected = ind(1:L);
            for j = 1 : L
                index = selected(j);
                s = Tr_all(:,index);
                shape = reshape(s,2,NUM_LANDMARKS);
                shape = shape + repmat( center, 1, NUM_LANDMARKS);
                init_shapes(j,:) = [ shape(1,:) shape(2,:) ];
            end
            arr_imgs_attri_test( map_test{m}(i) ).init_shapes.(train_para.shape_initialization) = init_shapes;

		end
	end
		
				
    % Save 
	arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
	arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
end

if strcmp( train_para.shape_initialization , 'modified_offcial_box_mean_shape_with_Gaussian_noise_level_0_05' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
            left = max( min(true_coords(1,:))-10, 1);
            up = max( min(true_coords(2,:))-10, 1);
            right = min( max(true_coords(1,:))+10, size(arr_imgs_attri(i).img,2) ); % right
            down = min( max(true_coords(2,:))+10, size(arr_imgs_attri(i).img,1) ); % down
            assert( right-left > 10); assert( down-up > 10 ); 
            box = [ left, up, right-left, down-up ];
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
    std_shapes_wrt_box  = mean( std(shapes_wrt_box) );
    noise_level = 0.05;
    samples = repmat( mean_shapes_wrt_box, N*L, 1 ) + random('norm',0,noise_level,N*L,2*NUM_LANDMARKS);
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
            left = max( min(true_coords(1,:))-10, 1);
            up = max( min(true_coords(2,:))-10, 1);
            right = min( max(true_coords(1,:))+10, size(arr_imgs_attri(i).img,2) ); % right
            down = min( max(true_coords(2,:))+10, size(arr_imgs_attri(i).img,1) ); % down
            assert( right-left > 10); assert( down-up > 10 ); 
            box = [ left, up, right-left, down-up ];
        else
            error('remain to be developed.');
        end
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, train_para.L_num_initial_shapes,1);
        init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:), repmat( box, train_para.L_num_initial_shapes,1) );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,4,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,4,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,4,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(3,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,4,4); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(5,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
         
        pTrue_test(i,:) = true_shape_row;
        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test(i,:) = init_shapes;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );

end

%% using offcial box + mean shape plus Gaussian noise
if strcmp( train_para.shape_initialization , 'official_box_mean_shape_with_Gaussian_noise_level_mean_Std' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
    std_shapes_wrt_box  = mean( std(shapes_wrt_box) );
    samples = repmat( mean_shapes_wrt_box, N*L, 1 ) + random('norm',0,std_shapes_wrt_box,N*L,2*NUM_LANDMARKS);
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, train_para.L_num_initial_shapes,1);
        init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:), repmat( box, train_para.L_num_initial_shapes,1) );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
         
        pTrue_test(i,:) = true_shape_row;
        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test(i,:) = init_shapes;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );

end


%% MTCNN_box_mean_shape_with_Gaussian_noise_level_mean_Std
if strcmp( train_para.shape_initialization , 'MTCNN_box_mean_shape_with_Gaussian_noise_level_mean_Std' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
    std_shapes_wrt_box  = mean( std(shapes_wrt_box) );
    samples = repmat( mean_shapes_wrt_box, N*L, 1 ) + random('norm',0,std_shapes_wrt_box,N*L,2*NUM_LANDMARKS);
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, train_para.L_num_initial_shapes,1);
        init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:), repmat( box, train_para.L_num_initial_shapes,1) );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
         
        pTrue_test(i,:) = true_shape_row;
        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test(i,:) = init_shapes;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end

%% mean_shape_with_Gaussian_sampling
if strcmp( train_para.shape_initialization , 'MTCNN_box_mean_shape_with_Multivariate_Normal_Sampling' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
    cov_shapes_wrt_box  = cov(shapes_wrt_box);
    samples = mvnrnd( mean_shapes_wrt_box, cov_shapes_wrt_box, length(arr_imgs_attri)*train_para.L_num_initial_shapes );
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, train_para.L_num_initial_shapes,1);
        init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:), repmat( box, train_para.L_num_initial_shapes,1) );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
         
        pTrue_test(i,:) = true_shape_row;
        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test(i,:) = init_shapes;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end

%% official_box_mean_shape_with_Multivariate_Normal_Sampling
if strcmp( train_para.shape_initialization , 'official_box_mean_shape_with_Multivariate_Normal_Sampling' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
    cov_shapes_wrt_box  = cov(shapes_wrt_box);
    samples = mvnrnd( mean_shapes_wrt_box, cov_shapes_wrt_box, length(arr_imgs_attri)*train_para.L_num_initial_shapes );
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, train_para.L_num_initial_shapes,1);
        init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:), repmat( box, train_para.L_num_initial_shapes,1) );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
         
        pTrue_test(i,:) = true_shape_row;
        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test(i,:) = init_shapes;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end

%% Simple connection between five points and all points
% Input: 5 landmarks estimated by MTCNN 
% Output: 20 landmarks
% Tr_s5  contains locations of 5 landmarks of all training samples
% Tr_all contains locations of NUM_LANDMARKS landmarks of all training samples
% Given the 5 landmarks of the test image, denoted by S5, we express it
% over Tr_s5 and obtain the sparse linear coefficients x, then the 29 landmarks can be estimated as Tr_all * x 
if strcmp( train_para.shape_initialization , 'simple_from_5_to_all' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
	arr_imgs_attri_train = arr_imgs_attri;

    Tr_s5 = zeros( 10, length(arr_imgs_attri_train) );
    Tr_all = zeros( NUM_LANDMARKS*2, length(arr_imgs_attri_train) );
    lambda = 0.005;
    L = train_para.L_num_initial_shapes;

    % Construct Tr_s5 and Tr_all
    for i = 1 : length( arr_imgs_attri_train )
        true_coords = arr_imgs_attri_train(i).true_coord_all_landmarks;
        five_points = arr_imgs_attri_train(i).coord_just_keypoints;
        center = mean( five_points, 2 );
        norm_five_points = five_points - repmat( center, 1, 5 );
        norm_five_points = reshape(norm_five_points,numel(norm_five_points),1);
        norm_five_points = norm_five_points / norm(norm_five_points);
        Tr_s5(:,i) = norm_five_points;
        norm_true_coords = true_coords - repmat( center, 1, NUM_LANDMARKS );
        norm_true_coords = reshape(norm_true_coords,numel(norm_true_coords),1);
        Tr_all(:,i) = norm_true_coords;
    end

    % Compute initial shapes for training samples
    for i = 1 : length( arr_imgs_attri_train )
        Tr_s5_bak = Tr_s5;
        Tr_all_bak = Tr_all;
        true_coords = arr_imgs_attri_train(i).true_coord_all_landmarks;
        true_coords_row = [true_coords(1,:) true_coords(2,:)];
        five_points = arr_imgs_attri_train(i).coord_just_keypoints;
        center = mean( five_points, 2 );
        norm_five_points = five_points - repmat( center, 1, 5 );
        norm_five_points = reshape(norm_five_points,numel(norm_five_points),1);
        norm_five_points = norm_five_points / norm(norm_five_points);
        Tr_s5_bak(:,i) = [];
        norm_true_coords = true_coords - repmat( center, 1, NUM_LANDMARKS );
        norm_true_coords = reshape(norm_true_coords,numel(norm_true_coords),1);
        Tr_all_bak(:,i) = [];

        % Repeat the representation for L times, to obtain L initial
        % shapes for every training sample
        partition = 0.3;
        init_shapes = zeros( L, 2*NUM_LANDMARKS);
        for j = 1 : L  
            selected = datasample(1:length(Tr_s5_bak), round(length(Tr_s5_bak)*partition), 'replace',false);
            sub_Tr_s5_bak = Tr_s5_bak(:,selected);
            sub_Tr_all_bak = Tr_all_bak(:,selected);
            fit = glmnet( sub_Tr_s5_bak, double(norm_five_points), 'gaussian', struct('lambda',lambda,'intr',0) );
            est = sub_Tr_all_bak * fit.beta;
            est_shape = reshape( est, 2, NUM_LANDMARKS );
            est_shape = est_shape + repmat( center, 1, NUM_LANDMARKS );
            est_shape_row = [est_shape(1,:) est_shape(2,:)];
            nme_train( (i-1)*L+j ) = shapeGt( 'dist', model , est_shape_row, true_coords_row );
            init_shapes(j,:) = est_shape_row;
        end
        arr_imgs_attri_train(i).init_shapes.(train_para.shape_initialization) = init_shapes;
    end
    % Save
	arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri','-v7.3' );
    fprintf('%s saved\n', Normalization_file );
    
    %%%%%%%%%%%%%%%% Generate initial shape for test samples, one initial
    %%%%%%%%%%%%%%%% shape per test sample
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    assert( exist(Normalization_file,'file') ~= 0 );
	load( Normalization_file );
	arr_imgs_attri_test = arr_imgs_attri;
    
    for i = 1 : length( arr_imgs_attri_test )
        item = arr_imgs_attri_test(i);
        true_coords = arr_imgs_attri_test(i).true_coord_all_landmarks;
        true_coords_row = [ true_coords(1,:) true_coords(2,:)];
        five_points = arr_imgs_attri_test(i).coord_just_keypoints;
        center = mean( five_points, 2 );
        norm_five_points = five_points - repmat( center, 1, 5 );
        norm_five_points = reshape(norm_five_points,numel(norm_five_points),1);
        norm_five_points = norm_five_points / norm(norm_five_points);
        
        % Express the norm_five_points over Tr_s5
        fit = glmnet( Tr_s5, double(norm_five_points), 'gaussian',struct('lambda',lambda,'intr',0));
        est = Tr_all * fit.beta;
        est_shape = reshape( est, 2 , NUM_LANDMARKS );
        est_shape = est_shape + repmat( center, 1, NUM_LANDMARKS );
        est_shape_row = [est_shape(1,:) est_shape(2,:)];
        nme_test(i) = shapeGt( 'dist', model , est_shape_row, true_coords_row );
        arr_imgs_attri_test(i).init_shapes.(train_para.shape_initialization) = est_shape_row;
    end
    % Save 
	arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
end

%% comparing_with_LBP with official box
if strcmp( train_para.shape_initialization , 'comparing_with_LBP_official_box' )
    Tr_LBP = []; Test_LBP = []; 
    %% Step 1: Store LBP features of every training sample as well as shapes
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    if ~exist('arr_imgs_attri_train','var')
        load( Normalization_file );
        arr_imgs_attri_train = arr_imgs_attri;
    end
    N = length(arr_imgs_attri);
    Dim_Feat = 8 * 8 * (8+2);
    % Column major
    Tr_LBP    = zeros( Dim_Feat, length(arr_imgs_attri_train) );
    % shape is normalized with respect to the box,i.e., box-normalized
    % shape
    Tr_shapes = zeros( 2*NUM_LANDMARKS, length(arr_imgs_attri_train) );
    for i = 1 : length(arr_imgs_attri_train)
        item = arr_imgs_attri_train(i);
        shape = item.true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri_train(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri_train(i).official_box;
        else
            error('remains to be developed\n');
        end
        
        % Crop the region inside the box
        img = item.img;
        left = box(1); up = box(2); right = left+box(3); down = up+box(4);
        region = img( up:down, left:right, : );
        if size(region==3)
            region = rgb2gray(region);
        end
        
        % Extract LBP features
        region_size = size( region );
        cell_size = floor( region_size / 8 );
        if min(min(cell_size)) < 5
            close all;figure(i); 
            subplot(1,2,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box)); imshow(tmp); title('true');
            subplot(1,2,2); tmp=disp_landmarks(item.img, item.coord_just_keypoints,struct('box',box)); imshow(tmp); title('true');
            de = 0;
        end
        Tr_LBP(:,i) = extractLBPFeatures( region,'cellsize',cell_size,'Upright',false);
        
        % Compute Box-normalized shape
        center = [ (left+right)/2 ; (up+down)/2 ];
        centered_shape = shape - repmat( center, 1, NUM_LANDMARKS);
        centered_shape(1,:) = centered_shape(1,:) / (box(3)/2) ;
        centered_shape(2,:) = centered_shape(2,:) / (box(4)/2) ;
        Tr_shapes(:,i) = reshape( centered_shape, numel(centered_shape), 1 );
    end
    
    %% Step 2: Select L most similar samples in the training set. 
    for i = 1 : length(arr_imgs_attri_train)
        item = arr_imgs_attri_train(i);
        true_shape = item.true_coord_all_landmarks;
        
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri_train(i).MTCNN_box;
            box_center = [box(1)+box(3)/2;box(2)+box(4)/2 ];
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri_train(i).official_box;
            box_center = [box(1)+box(3)/2;box(2)+box(4)/2 ];
        else
            error('remains to be developed\n');
        end
        cur_LBP = Tr_LBP(:,i);
        all_LBP = Tr_LBP;
        all_LBP(:,i) = []; % Remove the sample itself
        all_shapes = Tr_shapes;
        all_shapes(:,i) = []; % Remove the sample itself
        
        % Compute texture similarity between current face and Tr_LBP,using
        % Pearson similarity metric
        dis = zeros(length(all_LBP),1);
        for j = 1 : length(all_LBP)
            sim_matrix = corrcoef( cur_LBP, all_LBP(:,j) ); 
            dis(j) = sim_matrix(1,2);
        end
        
        % Select the L most similar shapes
        [sorted_data,ind] = sort(dis,'descend');
        ind = ind(1:L);
        init_shapes = zeros( L, 2*NUM_LANDMARKS );
        for j = 1 : L
            index = ind(j);
            % Obtain the corresponding box-normalized shape
            norm_shape = all_shapes(:,index);
            norm_shape = reshape( norm_shape, 2, NUM_LANDMARKS );
            % Transform to the current image
            norm_shape(1,:) = norm_shape(1,:) * ( box(3)/2 );
            norm_shape(2,:) = norm_shape(2,:) * ( box(4)/2 );
            norm_shape = norm_shape + repmat( box_center, 1, NUM_LANDMARKS );
            % Note: [x1 x2 .. xn; y1 y2 .. yn ]
            init_shapes(j,:) = [ norm_shape(1,:) norm_shape(2,:) ];
        end
        
        % Store
        arr_imgs_attri_train(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
%         % Visualization
%         close all; figure(i);
%         subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box) ); imshow(tmp); title('true');
%         subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:), struct('box',box) ); imshow(tmp); title('init shape');
%         subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:), struct('box',box) ); imshow(tmp); title('init shape');
    end
    % Save the result
    arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
    
    %% Select L most similar shapes for each test sample
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    if ~exist('arr_imgs_attri_test','var')
        load( Normalization_file );
        arr_imgs_attri_test = arr_imgs_attri;
    end
    for i = 1 : length(arr_imgs_attri_test)
        item = arr_imgs_attri_test(i);
        true_shape = item.true_coord_all_landmarks;
        
        if strcmp( train_para.box_type , 'MTCNN' )
            box = item.MTCNN_box;
            box_center = [box(1)+box(3)/2;box(2)+box(4)/2 ];
        elseif strcmp( train_para.box_type , 'official' )
            box = item.official_box;
            box_center = [box(1)+box(3)/2;box(2)+box(4)/2 ];
        else
            error('remains to be developed\n');
        end
        
        % Crop the region inside the box
        img = item.img;
        left = box(1); up = box(2); right = left+box(3); down = up+box(4);
        region = img( up:down, left:right, : );
        if size(region==3)
            region = rgb2gray(region);
        end
        
        % Extract LBP features
        region_size = size( region );
        cell_size = floor( region_size / 8 );
        cur_LBP = extractLBPFeatures( region,'cellsize',cell_size,'Upright',false);
        
        % Compute texture similarity between current face and Tr_LBP,using
        % Pearson similarity metric
        dis = zeros(length(Tr_LBP),1);
        for j = 1 : length(Tr_LBP)
            sim_matrix = corrcoef( cur_LBP, Tr_LBP(:,j) ); 
            dis(j) = sim_matrix(1,2);
        end
        
        % Select the L most similar shapes
        [sorted_data,ind] = sort(dis,'descend');
        ind = ind(1:L);
        init_shapes = zeros( L, 2*NUM_LANDMARKS );
        for j = 1 : L
            index = ind(j);
            % Obtain the corresponding box-normalized shape
            norm_shape = Tr_shapes(:,index);
            norm_shape = reshape( norm_shape, 2, NUM_LANDMARKS );
            % Transform to the current image
            norm_shape(1,:) = norm_shape(1,:) * ( box(3)/2 );
            norm_shape(2,:) = norm_shape(2,:) * ( box(4)/2 );
            norm_shape = norm_shape + repmat( box_center, 1, NUM_LANDMARKS );
            % Note: [x1 x2 .. xn; y1 y2 .. yn ]
            init_shapes(j,:) = [ norm_shape(1,:) norm_shape(2,:) ];
        end
        
        % Store
        arr_imgs_attri_test(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
%         % Visualization
%         close all; figure(i);
%         subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box) ); imshow(tmp); title('true');
%         subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:), struct('box',box) ); imshow(tmp); title('init shape');
%         subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:), struct('box',box) ); imshow(tmp); title('init shape');
    end
    % Save the result
    arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
end

%% comparing_with_LBP with MTCNN box
if strcmp( train_para.shape_initialization , 'comparing_with_LBP_MTCNN_box' )
    Tr_LBP = []; Test_LBP = []; 
    %% Step 1: Store LBP features of every training sample as well as shapes
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    if ~exist('arr_imgs_attri_train','var')
        load( Normalization_file );
        arr_imgs_attri_train = arr_imgs_attri;
    end
    N = length(arr_imgs_attri);
    Dim_Feat = 8 * 8 * (8+2);
    % Column major
    Tr_LBP    = zeros( Dim_Feat, length(arr_imgs_attri_train) );
    % shape is normalized with respect to the box,i.e., box-normalized
    % shape
    Tr_shapes = zeros( 2*NUM_LANDMARKS, length(arr_imgs_attri_train) );
    for i = 1 : length(arr_imgs_attri_train)
        item = arr_imgs_attri_train(i);
        shape = item.true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri_train(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri_train(i).official_box;
        else
            error('remains to be developed\n');
        end
        
        % Crop the region inside the box
        img = item.img;
        left = box(1); up = box(2); right = left+box(3); down = up+box(4);
        region = img( up:down, left:right, : );
        if size(region==3)
            region = rgb2gray(region);
        end
        
        % Extract LBP features
        region_size = size( region );
        cell_size = floor( region_size / 8 );
        if min(min(cell_size)) < 5
            close all;figure(i); 
            subplot(1,2,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box)); imshow(tmp); title('true');
            subplot(1,2,2); tmp=disp_landmarks(item.img, item.coord_just_keypoints,struct('box',box)); imshow(tmp); title('true');
            de = 0;
        end
        Tr_LBP(:,i) = extractLBPFeatures( region,'cellsize',cell_size,'Upright',false);
        
        % Compute Box-normalized shape
        center = [ (left+right)/2 ; (up+down)/2 ];
        centered_shape = shape - repmat( center, 1, NUM_LANDMARKS);
        centered_shape(1,:) = centered_shape(1,:) / (box(3)/2) ;
        centered_shape(2,:) = centered_shape(2,:) / (box(4)/2) ;
        Tr_shapes(:,i) = reshape( centered_shape, numel(centered_shape), 1 );
    end
    
    %% Step 2: Select L most similar samples in the training set. 
    for i = 1 : length(arr_imgs_attri_train)
        item = arr_imgs_attri_train(i);
        true_shape = item.true_coord_all_landmarks;
        
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri_train(i).MTCNN_box;
            box_center = [box(1)+box(3)/2;box(2)+box(4)/2 ];
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri_train(i).official_box;
            box_center = [box(1)+box(3)/2;box(2)+box(4)/2 ];
        else
            error('remains to be developed\n');
        end
        cur_LBP = Tr_LBP(:,i);
        all_LBP = Tr_LBP;
        all_LBP(:,i) = []; % Remove the sample itself
        all_shapes = Tr_shapes;
        all_shapes(:,i) = []; % Remove the sample itself
        
        % Compute texture similarity between current face and Tr_LBP,using
        % Pearson similarity metric
        dis = zeros(length(all_LBP),1);
        for j = 1 : length(all_LBP)
            sim_matrix = corrcoef( cur_LBP, all_LBP(:,j) ); 
            dis(j) = sim_matrix(1,2);
        end
        
        % Select the L most similar shapes
        [sorted_data,ind] = sort(dis,'descend');
        ind = ind(1:L);
        init_shapes = zeros( L, 2*NUM_LANDMARKS );
        for j = 1 : L
            index = ind(j);
            % Obtain the corresponding box-normalized shape
            norm_shape = all_shapes(:,index);
            norm_shape = reshape( norm_shape, 2, NUM_LANDMARKS );
            % Transform to the current image
            norm_shape(1,:) = norm_shape(1,:) * ( box(3)/2 );
            norm_shape(2,:) = norm_shape(2,:) * ( box(4)/2 );
            norm_shape = norm_shape + repmat( box_center, 1, NUM_LANDMARKS );
            % Note: [x1 x2 .. xn; y1 y2 .. yn ]
            init_shapes(j,:) = [ norm_shape(1,:) norm_shape(2,:) ];
        end
        
        % Store
        arr_imgs_attri_train(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
%         % Visualization
%         close all; figure(i);
%         subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box) ); imshow(tmp); title('true');
%         subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:), struct('box',box) ); imshow(tmp); title('init shape');
%         subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:), struct('box',box) ); imshow(tmp); title('init shape');
    end
    % Save the result
    arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
    
    %% Select L most similar shapes for each test sample
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    if ~exist('arr_imgs_attri_test','var')
        load( Normalization_file );
        arr_imgs_attri_test = arr_imgs_attri;
    end
    for i = 1 : length(arr_imgs_attri_test)
        item = arr_imgs_attri_test(i);
        true_shape = item.true_coord_all_landmarks;
        
        if strcmp( train_para.box_type , 'MTCNN' )
            box = item.MTCNN_box;
            box_center = [box(1)+box(3)/2;box(2)+box(4)/2 ];
        elseif strcmp( train_para.box_type , 'official' )
            box = item.official_box;
            box_center = [box(1)+box(3)/2;box(2)+box(4)/2 ];
        else
            error('remains to be developed\n');
        end
        
        % Crop the region inside the box
        img = item.img;
        left = box(1); up = box(2); right = left+box(3); down = up+box(4);
        region = img( up:down, left:right, : );
        if size(region==3)
            region = rgb2gray(region);
        end
        
        % Extract LBP features
        region_size = size( region );
        cell_size = floor( region_size / 8 );
        cur_LBP = extractLBPFeatures( region,'cellsize',cell_size,'Upright',false);
        
        % Compute texture similarity between current face and Tr_LBP,using
        % Pearson similarity metric
        dis = zeros(length(Tr_LBP),1);
        for j = 1 : length(Tr_LBP)
            sim_matrix = corrcoef( cur_LBP, Tr_LBP(:,j) ); 
            dis(j) = sim_matrix(1,2);
        end
        
        % Select the L most similar shapes
        [sorted_data,ind] = sort(dis,'descend');
        ind = ind(1:L);
        init_shapes = zeros( L, 2*NUM_LANDMARKS );
        for j = 1 : L
            index = ind(j);
            % Obtain the corresponding box-normalized shape
            norm_shape = Tr_shapes(:,index);
            norm_shape = reshape( norm_shape, 2, NUM_LANDMARKS );
            % Transform to the current image
            norm_shape(1,:) = norm_shape(1,:) * ( box(3)/2 );
            norm_shape(2,:) = norm_shape(2,:) * ( box(4)/2 );
            norm_shape = norm_shape + repmat( box_center, 1, NUM_LANDMARKS );
            % Note: [x1 x2 .. xn; y1 y2 .. yn ]
            init_shapes(j,:) = [ norm_shape(1,:) norm_shape(2,:) ];
        end
        
        % Store
        arr_imgs_attri_test(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
%         % Visualization
%         close all; figure(i);
%         subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box) ); imshow(tmp); title('true');
%         subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:), struct('box',box) ); imshow(tmp); title('init shape');
%         subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:), struct('box',box) ); imshow(tmp); title('init shape');
    end
    % Save the result
    arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
end

%% LBP similarity based on cropped MTCNN box
if strcmp( train_para.shape_initialization , 'LBP_simi_Pearson_crop_MTCNN_keypoints' )
    Tr_LBP = []; Test_LBP = []; 
    %% Step 1: Store LBP features of every training sample as well as shapes
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    if ~exist('arr_imgs_attri_train','var')
        load( Normalization_file );
        arr_imgs_attri_train = arr_imgs_attri;
    end
    N = length(arr_imgs_attri);
    Dim_Feat = 8 * 8 * (8+2);
    % Column major
    Tr_LBP    = zeros( Dim_Feat, length(arr_imgs_attri_train) );
    % shape is normalized with respect to the box,i.e., box-normalized
    % shape
    Tr_shapes = zeros( 2*NUM_LANDMARKS, length(arr_imgs_attri_train) );
    for i = 1 : length(arr_imgs_attri_train)
        item = arr_imgs_attri_train(i);
        shape = item.true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        
        % Crop the region inside the box
        img = item.img;
        left = box(1); up = box(2); right = left+box(3); down = up+box(4);
        region = img( up:down, left:right, : );
        if size(region==3)
            region = rgb2gray(region);
        end
        
        % Extract LBP features
        region_size = size( region );
        cell_size = floor( region_size / 8 );
        if min(min(cell_size)) < 5
            close all;figure(i); 
            subplot(1,2,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box)); imshow(tmp); title('true');
            subplot(1,2,2); tmp=disp_landmarks(item.img, item.coord_just_keypoints,struct('box',box)); imshow(tmp); title('true');
            error('invalid');
        end
        Tr_LBP(:,i) = extractLBPFeatures( region,'cellsize',cell_size,'Upright',false);
        
        % Compute Box-normalized shape
        center = [ (left+right)/2 ; (up+down)/2 ];
        centered_shape = shape - repmat( center, 1, NUM_LANDMARKS);
        centered_shape(1,:) = centered_shape(1,:) / (box(3)/2) ;
        centered_shape(2,:) = centered_shape(2,:) / (box(4)/2) ;
        Tr_shapes(:,i) = reshape( centered_shape, numel(centered_shape), 1 );
    end
    
    %% Step 2: Select L most similar samples in the training set. 
    for i = 1 : length(arr_imgs_attri_train)
        item = arr_imgs_attri_train(i);
        true_shape = item.true_coord_all_landmarks;
        
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        box_center = [ box(1)+box(3)/2 ; box(2)+box(4)/2 ];
        cur_LBP = Tr_LBP(:,i);
        all_LBP = Tr_LBP;
        all_LBP(:,i) = []; % Remove the sample itself
        all_shapes = Tr_shapes;
        all_shapes(:,i) = []; % Remove the sample itself
        
        % Compute texture similarity between current face and Tr_LBP,using
        % Pearson similarity metric
        dis = zeros(length(all_LBP),1);
        for j = 1 : length(all_LBP)
            sim_matrix = corrcoef( cur_LBP, all_LBP(:,j) ); 
            dis(j) = sim_matrix(1,2);
        end
        
        % Select the L most similar shapes
        [sorted_data,ind] = sort(dis,'descend');
        ind = ind(1:L);
        init_shapes = zeros( L, 2*NUM_LANDMARKS );
        for j = 1 : L
            index = ind(j);
            % Obtain the corresponding box-normalized shape
            norm_shape = all_shapes(:,index);
            norm_shape = reshape( norm_shape, 2, NUM_LANDMARKS );
            % Transform to the current image
            norm_shape(1,:) = norm_shape(1,:) * ( box(3)/2 );
            norm_shape(2,:) = norm_shape(2,:) * ( box(4)/2 );
            norm_shape = norm_shape + repmat( box_center, 1, NUM_LANDMARKS );
            % Note: [x1 x2 .. xn; y1 y2 .. yn ]
            init_shapes(j,:) = [ norm_shape(1,:) norm_shape(2,:) ];
        end
        
        % Store
        arr_imgs_attri_train(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
%         % Visualization
%         close all; figure(i);
%         subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box) ); imshow(tmp); title('true');
%         subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:), struct('box',box) ); imshow(tmp); title('init shape');
%         subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:), struct('box',box) ); imshow(tmp); title('init shape');
    end
    % Save the result
    arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
    
    %% Select L most similar shapes for each test sample
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    if ~exist('arr_imgs_attri_test','var')
        load( Normalization_file );
        arr_imgs_attri_test = arr_imgs_attri;
    end
    for i = 1 : length(arr_imgs_attri_test)
        item = arr_imgs_attri_test(i);
        true_shape = item.true_coord_all_landmarks;
        
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        box_center = [ box(1)+box(3)/2 ; box(2)+box(4)/2 ];
        
        % Crop the region inside the box
        img = item.img;
        left = box(1); up = box(2); right = left+box(3); down = up+box(4);
        region = img( up:down, left:right, : );
        if size(region==3)
            region = rgb2gray(region);
        end
        
        % Extract LBP features
        region_size = size( region );
        cell_size = floor( region_size / 8 );
        cur_LBP = extractLBPFeatures( region,'cellsize',cell_size,'Upright',false);
        
        % Compute texture similarity between current face and Tr_LBP,using
        % Pearson similarity metric
        dis = zeros(length(Tr_LBP),1);
        for j = 1 : length(Tr_LBP)
            sim_matrix = corrcoef( cur_LBP, Tr_LBP(:,j) ); 
            dis(j) = sim_matrix(1,2);
        end
        
        % Select the L most similar shapes
        [sorted_data,ind] = sort(dis,'descend');
        ind = ind(1:L);
        init_shapes = zeros( L, 2*NUM_LANDMARKS );
        for j = 1 : L
            index = ind(j);
            % Obtain the corresponding box-normalized shape
            norm_shape = Tr_shapes(:,index);
            norm_shape = reshape( norm_shape, 2, NUM_LANDMARKS );
            % Transform to the current image
            norm_shape(1,:) = norm_shape(1,:) * ( box(3)/2 );
            norm_shape(2,:) = norm_shape(2,:) * ( box(4)/2 );
            norm_shape = norm_shape + repmat( box_center, 1, NUM_LANDMARKS );
            % Note: [x1 x2 .. xn; y1 y2 .. yn ]
            init_shapes(j,:) = [ norm_shape(1,:) norm_shape(2,:) ];
        end
        
        % Store
        arr_imgs_attri_test(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
%         % Visualization
%         close all; figure(i);
%         subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box) ); imshow(tmp); title('true');
%         subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:), struct('box',box) ); imshow(tmp); title('init shape');
%         subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:), struct('box',box) ); imshow(tmp); title('init shape');
    end
    % Save the result
    arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
end

%% LBP similarity based on cropped MTCNN box
if strcmp( train_para.shape_initialization , 'HOG_of_MTCNN_keypoints_ls' )
    Tr_LBP = []; Test_LBP = []; 
    %% Step 1: Store LBP features of every training sample as well as shapes
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    if ~exist('arr_imgs_attri_train','var')
        load( Normalization_file );
        arr_imgs_attri_train = arr_imgs_attri;
    end
    N = length(arr_imgs_attri);
    Dim_Feat = 8 * 8 * (8+2);
    % Column major
    Tr_LBP    = zeros( Dim_Feat, length(arr_imgs_attri_train) );
    % shape is normalized with respect to the box,i.e., box-normalized
    % shape
    Tr_shapes = zeros( 2*NUM_LANDMARKS, length(arr_imgs_attri_train) );
    for i = 1 : length(arr_imgs_attri_train)
        item = arr_imgs_attri_train(i);
        shape = item.true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        
        % Crop the region inside the box
        img = item.img;
        left = box(1); up = box(2); right = left+box(3); down = up+box(4);
        region = img( up:down, left:right, : );
        if size(region==3)
            region = rgb2gray(region);
        end
        
        % Extract LBP features
        region_size = size( region );
        cell_size = floor( region_size / 8 );
        if min(min(cell_size)) < 5
            close all;figure(i); 
            subplot(1,2,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box)); imshow(tmp); title('true');
            subplot(1,2,2); tmp=disp_landmarks(item.img, item.coord_just_keypoints,struct('box',box)); imshow(tmp); title('true');
            error('invalid');
        end
        Tr_LBP(:,i) = extractLBPFeatures( region,'cellsize',cell_size,'Upright',false);
        
        % Compute Box-normalized shape
        center = [ (left+right)/2 ; (up+down)/2 ];
        centered_shape = shape - repmat( center, 1, NUM_LANDMARKS);
        centered_shape(1,:) = centered_shape(1,:) / (box(3)/2) ;
        centered_shape(2,:) = centered_shape(2,:) / (box(4)/2) ;
        Tr_shapes(:,i) = reshape( centered_shape, numel(centered_shape), 1 );
    end
    
    %% Step 2: Select L most similar samples in the training set. 
    for i = 1 : length(arr_imgs_attri_train)
        item = arr_imgs_attri_train(i);
        true_shape = item.true_coord_all_landmarks;
        
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        box_center = [ box(1)+box(3)/2 ; box(2)+box(4)/2 ];
        cur_LBP = Tr_LBP(:,i);
        all_LBP = Tr_LBP;
        all_LBP(:,i) = []; % Remove the sample itself
        all_shapes = Tr_shapes;
        all_shapes(:,i) = []; % Remove the sample itself
        
        % Compute texture similarity between current face and Tr_LBP,using
        % Pearson similarity metric
        dis = zeros(length(all_LBP),1);
        for j = 1 : length(all_LBP)
            sim_matrix = corrcoef( cur_LBP, all_LBP(:,j) ); 
            dis(j) = sim_matrix(1,2);
        end
        
        % Select the L most similar shapes
        [sorted_data,ind] = sort(dis,'descend');
        ind = ind(1:L);
        init_shapes = zeros( L, 2*NUM_LANDMARKS );
        for j = 1 : L
            index = ind(j);
            % Obtain the corresponding box-normalized shape
            norm_shape = all_shapes(:,index);
            norm_shape = reshape( norm_shape, 2, NUM_LANDMARKS );
            % Transform to the current image
            norm_shape(1,:) = norm_shape(1,:) * ( box(3)/2 );
            norm_shape(2,:) = norm_shape(2,:) * ( box(4)/2 );
            norm_shape = norm_shape + repmat( box_center, 1, NUM_LANDMARKS );
            % Note: [x1 x2 .. xn; y1 y2 .. yn ]
            init_shapes(j,:) = [ norm_shape(1,:) norm_shape(2,:) ];
        end
        
        % Store
        arr_imgs_attri_train(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
%         % Visualization
%         close all; figure(i);
%         subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box) ); imshow(tmp); title('true');
%         subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:), struct('box',box) ); imshow(tmp); title('init shape');
%         subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:), struct('box',box) ); imshow(tmp); title('init shape');
    end
    % Save the result
    arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
    
    %% Select L most similar shapes for each test sample
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    if ~exist('arr_imgs_attri_test','var')
        load( Normalization_file );
        arr_imgs_attri_test = arr_imgs_attri;
    end
    for i = 1 : length(arr_imgs_attri_test)
        item = arr_imgs_attri_test(i);
        true_shape = item.true_coord_all_landmarks;
        
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        box_center = [ box(1)+box(3)/2 ; box(2)+box(4)/2 ];
        
        % Crop the region inside the box
        img = item.img;
        left = box(1); up = box(2); right = left+box(3); down = up+box(4);
        region = img( up:down, left:right, : );
        if size(region==3)
            region = rgb2gray(region);
        end
        
        % Extract LBP features
        region_size = size( region );
        cell_size = floor( region_size / 8 );
        cur_LBP = extractLBPFeatures( region,'cellsize',cell_size,'Upright',false);
        
        % Compute texture similarity between current face and Tr_LBP,using
        % Pearson similarity metric
        dis = zeros(length(Tr_LBP),1);
        for j = 1 : length(Tr_LBP)
            sim_matrix = corrcoef( cur_LBP, Tr_LBP(:,j) ); 
            dis(j) = sim_matrix(1,2);
        end
        
        % Select the L most similar shapes
        [sorted_data,ind] = sort(dis,'descend');
        ind = ind(1:L);
        init_shapes = zeros( L, 2*NUM_LANDMARKS );
        for j = 1 : L
            index = ind(j);
            % Obtain the corresponding box-normalized shape
            norm_shape = Tr_shapes(:,index);
            norm_shape = reshape( norm_shape, 2, NUM_LANDMARKS );
            % Transform to the current image
            norm_shape(1,:) = norm_shape(1,:) * ( box(3)/2 );
            norm_shape(2,:) = norm_shape(2,:) * ( box(4)/2 );
            norm_shape = norm_shape + repmat( box_center, 1, NUM_LANDMARKS );
            % Note: [x1 x2 .. xn; y1 y2 .. yn ]
            init_shapes(j,:) = [ norm_shape(1,:) norm_shape(2,:) ];
        end
        
        % Store
        arr_imgs_attri_test(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
%         % Visualization
%         close all; figure(i);
%         subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,struct('box',box) ); imshow(tmp); title('true');
%         subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:), struct('box',box) ); imshow(tmp); title('init shape');
%         subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:), struct('box',box) ); imshow(tmp); title('init shape');
    end
    % Save the result
    arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri' );
    fprintf('%s saved\n', Normalization_file );
end

%% mean_shape_with_box_augmentation
if strcmp( train_para.shape_initialization , 'mean_shape_with_box_augmentation_MTCNN' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
    samples = kron(mean_shapes_wrt_box,ones(N*L,1));
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        img = arr_imgs_attri(i).img; w = size(img,2); h = size(img,1); assert(w>0); assert(h>0);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, L,1);
        
        %% Perform box augmentation
        % Denote the upper-left and lower-right corners
        box_corner = [ box(1) box(2) box(1)+box(3) box(2)+box(4) ]; % [x1 y1 x2 y2 ]
        box_corner = repmat( box_corner, L, 1 );
        % Add uniform noise to box corners, 10% * box_width, 10% *
        % box_height
        Noise_Level = 0.1; 
        x_range = box(3) * Noise_Level;
        y_range = box(4) * Noise_Level;
        box_corner(:,1) = box_corner(:,1) + random('uniform', -x_range, x_range, L, 1 );
        box_corner(:,2) = box_corner(:,2) + random('uniform', -y_range, y_range, L, 1 );
        box_corner(:,3) = box_corner(:,3) + random('uniform', -x_range, x_range, L, 1 );
        box_corner(:,4) = box_corner(:,4) + random('uniform', -y_range, y_range, L, 1 );
        % have a check, deal with exceptions: out of the face image
        x1=box_corner(:,1); y1=box_corner(:,2); x2=box_corner(:,3); y2=box_corner(:,4);
        % x1 
        x1( x1 < 1 ) = 1; x1( x1 > w ) = w;
        % y1 
        y1( y1 < 1 ) = 1; y1( y1 > h ) = h;
        % x2
        x2( x2 < 1 ) = 1; x2( x2 > w ) = w;
        % y2 
        y2( y2 < 1 ) = 1; y2( y2 > h ) = h;
        box_corner = [ x1 y1 x2 y2 ];
        % compute augmented boxes
        boxes = [ box_corner(:,1) box_corner(:,2) box_corner(:,3)-box_corner(:,1) box_corner(:,4)-box_corner(:,2) ];     
        
        init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*L+1): (i*L),:), boxes );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit( ((i-1)*L+1): (i*L),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',boxes(1,:) ));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',boxes(L,:) ));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
         
        pTrue_test(i,:) = true_shape_row;
        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test(i,:) = init_shapes;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end

%% mean_shape_with_box_augmentation_official
if strcmp( train_para.shape_initialization , 'mean_shape_with_box_augmentation_official' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
    cov_shapes_wrt_box  = cov(shapes_wrt_box);
    samples = mvnrnd( mean_shapes_wrt_box, cov_shapes_wrt_box, length(arr_imgs_attri)*train_para.L_num_initial_shapes );
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, train_para.L_num_initial_shapes,1);
        init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:), repmat( box, train_para.L_num_initial_shapes,1) );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        else
            error('remain to be developed.');
        end
         
        pTrue_test(i,:) = true_shape_row;
        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test(i,:) = init_shapes;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end

%% mean_shape_with_box_augmentation_official
if strcmp( train_para.shape_initialization , 'mean_shape_with_box_aug_crop_MTCNN_keypoints_0_2' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
    samples = repmat( mean_shapes_wrt_box, length(arr_imgs_attri)*train_para.L_num_initial_shapes, 1 );
       
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        w = size(item.img,2); h = size(item.img,1);
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        %% Perform box augmentation
        % Denote the upper-left and lower-right corners
        box_corner = [ box(1) box(2) box(1)+box(3) box(2)+box(4) ]; % [x1 y1 x2 y2 ]
        box_corner = repmat( box_corner, L, 1 );
        % Add uniform noise to box corners, 10% * box_width, 10% *
        % box_height
        Noise_Level = 0.2; 
        x_range = box(3) * Noise_Level;
        y_range = box(4) * Noise_Level;
        box_corner(:,1) = box_corner(:,1) + random('uniform', -x_range, x_range, L, 1 );
        box_corner(:,2) = box_corner(:,2) + random('uniform', -y_range, y_range, L, 1 );
        box_corner(:,3) = box_corner(:,3) + random('uniform', -x_range, x_range, L, 1 );
        box_corner(:,4) = box_corner(:,4) + random('uniform', -y_range, y_range, L, 1 );
        % have a check, deal with exceptions: out of the face image
        x1=box_corner(:,1); y1=box_corner(:,2); x2=box_corner(:,3); y2=box_corner(:,4);
        % x1 
        x1( x1 < 1 ) = 1; x1( x1 > w ) = w;
        % y1 
        y1( y1 < 1 ) = 1; y1( y1 > h ) = h;
        % x2
        x2( x2 < 1 ) = 1; x2( x2 > w ) = w;
        % y2 
        y2( y2 < 1 ) = 1; y2( y2 > h ) = h;
        box_corner = [ x1 y1 x2 y2 ];
        % compute augmented boxes
        boxes = [ box_corner(:,1) box_corner(:,2) box_corner(:,3)-box_corner(:,1) box_corner(:,4)-box_corner(:,2) ];     
        
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, train_para.L_num_initial_shapes,1);
        init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:), boxes );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
         
        pTrue_test(i,:) = true_shape_row;
        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test(i,:) = init_shapes;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end

%% mean_shape_with_box_aug_crop_MTCNN_keypoints_with_pose
if strcmp( train_para.shape_initialization , 'mean_shape_with_box_aug_crop_MTCNN_keypoints_poses' )
    
    if ~exist('arr_imgs_attri_train')
        Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
        load( Normalization_file );
        arr_imgs_attri_train = arr_imgs_attri;
    end
    if ~exist('arr_imgs_attri_test')
        Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
        load( Normalization_file );
        arr_imgs_attri_test = arr_imgs_attri;
    end
    L = train_para.L_num_initial_shapes;
    
    %%% Note: Divide three poses !
    load( ['map_poses_',DATA_SET,'.mat'] );
    load( ['map_test_poses_',DATA_SET,'.mat'] );
    assert( length(map{1})>0 & length(map{2})>0  & length(map{3})>0  );
    assert( length(map_test{1})>0 & length(map_test{2})>0 & length(map_test{3})>0  );

    for m = 1 : 3
        
        % % % Get shapes w.r.t MTCNN box of all training samples
        shapes_wrt_box = zeros( length(map{m}), 2*NUM_LANDMARKS );
        for i = 1 : length( map{m} )
            item = arr_imgs_attri_train( map{m}(i) );
            true_coords = item.true_coord_all_landmarks;
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
                MTCNN_keypoints = item.coord_just_keypoints;
                x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
                y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            else
                error('remain to be developed.');
            end
            norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
            norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
            shapes_wrt_box(i,:) = norm_shape;
        end
        % Build the multivariate normal distribution model for
        % shapes_wrt_box
        mean_shapes_wrt_box = mean( shapes_wrt_box );
        samples = repmat( mean_shapes_wrt_box, length(map{m})*L, 1 );
       
        % Generate initial shapes for each training sample( raw image
        % coords )
        pTrue = zeros( length(map{m})*L, 2*NUM_LANDMARKS);
        pInit = zeros( length(map{m})*L, 2*NUM_LANDMARKS);
        for i = 1 : length(map{m})
            item = arr_imgs_attri_train( map{m}(i) );
            w = size(item.img,2); h = size(item.img,1);
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
                MTCNN_keypoints = item.coord_just_keypoints;
                x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
                y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            else
                error('remain to be developed.');
            end
            %% Perform box augmentation
            % Denote the upper-left and lower-right corners
            box_corner = [ box(1) box(2) box(1)+box(3) box(2)+box(4) ]; % [x1 y1 x2 y2 ]
            box_corner = repmat( box_corner, L, 1 );
            % Add uniform noise to box corners, 10% * box_width, 10% *
            % box_height
            Noise_Level = 0.1; 
            x_range = box(3) * Noise_Level;
            y_range = box(4) * Noise_Level;
            box_corner(:,1) = box_corner(:,1) + random('uniform', -x_range, x_range, L, 1 );
            box_corner(:,2) = box_corner(:,2) + random('uniform', -y_range, y_range, L, 1 );
            box_corner(:,3) = box_corner(:,3) + random('uniform', -x_range, x_range, L, 1 );
            box_corner(:,4) = box_corner(:,4) + random('uniform', -y_range, y_range, L, 1 );
            % have a check, deal with exceptions: out of the face image
            x1=box_corner(:,1); y1=box_corner(:,2); x2=box_corner(:,3); y2=box_corner(:,4);
            % x1 
            x1( x1 < 1 ) = 1; x1( x1 > w ) = w;
            % y1 
            y1( y1 < 1 ) = 1; y1( y1 > h ) = h;
            % x2
            x2( x2 < 1 ) = 1; x2( x2 > w ) = w;
            % y2 
            y2( y2 < 1 ) = 1; y2( y2 > h ) = h;
            box_corner = [ x1 y1 x2 y2 ];
            % compute augmented boxes
            boxes = [ box_corner(:,1) box_corner(:,2) box_corner(:,3)-box_corner(:,1) box_corner(:,4)-box_corner(:,2) ];     

            true_shape = item.true_coord_all_landmarks;
            true_shape_row = [true_shape(1,:) true_shape(2,:)];
            pTrue( ((i-1)*L+1): (i*L),:) = repmat( true_shape_row, L,1);
            init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*L+1): (i*L),:), boxes );
            arr_imgs_attri_train( map{m}(i) ).init_shapes.(train_para.shape_initialization) = init_shapes;
            pInit( ((i-1)*L+1): (i*L),:) = init_shapes;
                % debugging : display
%                 close all; figure(i);
%                 subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,  struct('box',box));imshow(tmp);
%                 subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:),               struct('box',boxes(1,:)));imshow(tmp);
%                 subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:),               struct('box',boxes(L,:)));imshow(tmp);
%                 de = 0;
        end
        % Report NME
        nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
        fprintf('Report NME for Train: %.2f %%  m = %d\n', mean(nme_train)*100 , m );
    
        % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
        pTrue_test = zeros( length(map_test{m}), 2*NUM_LANDMARKS);
        pInit_test = zeros( length(map_test{m}), 2*NUM_LANDMARKS);
        for i = 1 : length(map_test{m})
            item = arr_imgs_attri_test( map_test{m}(i) );
            true_shape = item.true_coord_all_landmarks;
            true_shape_row = [true_shape(1,:) true_shape(2,:)];
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
                MTCNN_keypoints = item.coord_just_keypoints;
                x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
                y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            else
                error('remain to be developed.');
            end

            pTrue_test(i,:) = true_shape_row;
            init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
            arr_imgs_attri_test( map_test{m}(i) ).init_shapes.(train_para.shape_initialization) = init_shapes;
            pInit_test(i,:) = init_shapes;
    %         % debugging : display
    %         close all; figure(i);
    %         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
    %         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
    % 
    %         de = 0;
        end
        % Report NME
        nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
        fprintf('Report NME for Test: %.2f %%  m = %d\n', mean(nme_test)*100 , m );
    end
    
    % % % Save to file
    arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n', Normalization_file );
        
    arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n', Normalization_file );
    

end

%% mean_shape_with_box_aug_crop_MTCNN_keypoints_with_pose
if strcmp( train_para.shape_initialization , 'mean_shape_with_box_aug_crop_true_keypoints_poses' )
    
    if ~exist('arr_imgs_attri_train')
        Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
        load( Normalization_file );
        arr_imgs_attri_train = arr_imgs_attri;
    end
    if ~exist('arr_imgs_attri_test')
        Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
        load( Normalization_file );
        arr_imgs_attri_test = arr_imgs_attri;
    end
    L = train_para.L_num_initial_shapes;
    
    %%% Note: Divide three poses !
    load( ['map_poses_',DATA_SET,'.mat'] );
    load( ['map_test_poses_',DATA_SET,'.mat'] );
    assert( length(map{1})>0 & length(map{2})>0  & length(map{3})>0  );
    assert( length(map_test{1})>0 & length(map_test{2})>0 & length(map_test{3})>0  );

    for m = 1 : 3
        
        % % % Get shapes w.r.t MTCNN box of all training samples
        shapes_wrt_box = zeros( length(map{m}), 2*NUM_LANDMARKS );
        for i = 1 : length( map{m} )
            item = arr_imgs_attri_train( map{m}(i) );
            true_coords = item.true_coord_all_landmarks;
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
                MTCNN_keypoints = item.coord_just_keypoints;
                x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
                y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            elseif strcmp( train_para.box_type,'crop_true_keypoints' )
                true_keypoints = item.true_coord_all_landmarks;
                assert( NUM_LANDMARKS == 29 );
                true_keypoints = true_keypoints(:,[17 18 21 23 24] );
                x_min = min(true_keypoints(1,:)); x_max = max(true_keypoints(1,:));
                y_min = min(true_keypoints(2,:)); y_max = max(true_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            else
                error('remain to be developed.');
            end
            norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
            norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
            shapes_wrt_box(i,:) = norm_shape;
        end
        % Build the multivariate normal distribution model for
        % shapes_wrt_box
        mean_shapes_wrt_box = mean( shapes_wrt_box );
        samples = repmat( mean_shapes_wrt_box, length(map{m})*L, 1 );
       
        % Generate initial shapes for each training sample( raw image
        % coords )
        pTrue = zeros( length(map{m})*L, 2*NUM_LANDMARKS);
        pInit = zeros( length(map{m})*L, 2*NUM_LANDMARKS);
        for i = 1 : length(map{m})
            item = arr_imgs_attri_train( map{m}(i) );
            w = size(item.img,2); h = size(item.img,1);
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
                MTCNN_keypoints = item.coord_just_keypoints;
                x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
                y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            elseif strcmp( train_para.box_type,'crop_true_keypoints' )
                true_keypoints = item.true_coord_all_landmarks;
                assert( NUM_LANDMARKS == 29 );
                true_keypoints = true_keypoints(:,[17 18 21 23 24] );
                x_min = min(true_keypoints(1,:)); x_max = max(true_keypoints(1,:));
                y_min = min(true_keypoints(2,:)); y_max = max(true_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            else
                error('remain to be developed.');
            end
            %% Perform box augmentation
            % Denote the upper-left and lower-right corners
            box_corner = [ box(1) box(2) box(1)+box(3) box(2)+box(4) ]; % [x1 y1 x2 y2 ]
            box_corner = repmat( box_corner, L, 1 );
            % Add uniform noise to box corners, 10% * box_width, 10% *
            % box_height
            Noise_Level = 0.1; 
            x_range = box(3) * Noise_Level;
            y_range = box(4) * Noise_Level;
            box_corner(:,1) = box_corner(:,1) + random('uniform', -x_range, x_range, L, 1 );
            box_corner(:,2) = box_corner(:,2) + random('uniform', -y_range, y_range, L, 1 );
            box_corner(:,3) = box_corner(:,3) + random('uniform', -x_range, x_range, L, 1 );
            box_corner(:,4) = box_corner(:,4) + random('uniform', -y_range, y_range, L, 1 );
            % have a check, deal with exceptions: out of the face image
            x1=box_corner(:,1); y1=box_corner(:,2); x2=box_corner(:,3); y2=box_corner(:,4);
            % x1 
            x1( x1 < 1 ) = 1; x1( x1 > w ) = w;
            % y1 
            y1( y1 < 1 ) = 1; y1( y1 > h ) = h;
            % x2
            x2( x2 < 1 ) = 1; x2( x2 > w ) = w;
            % y2 
            y2( y2 < 1 ) = 1; y2( y2 > h ) = h;
            box_corner = [ x1 y1 x2 y2 ];
            % compute augmented boxes
            boxes = [ box_corner(:,1) box_corner(:,2) box_corner(:,3)-box_corner(:,1) box_corner(:,4)-box_corner(:,2) ];     

            true_shape = item.true_coord_all_landmarks;
            true_shape_row = [true_shape(1,:) true_shape(2,:)];
            pTrue( ((i-1)*L+1): (i*L),:) = repmat( true_shape_row, L,1);
            init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*L+1): (i*L),:), boxes );
            arr_imgs_attri_train( map{m}(i) ).init_shapes.(train_para.shape_initialization) = init_shapes;
            pInit( ((i-1)*L+1): (i*L),:) = init_shapes;
                % debugging : display
%                 close all; figure(i);
%                 subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,  struct('box',box));imshow(tmp);
%                 subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:),               struct('box',boxes(1,:)));imshow(tmp);
%                 subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:),               struct('box',boxes(L,:)));imshow(tmp);
%                 de = 0;
        end
        % Report NME
        nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
        fprintf('Report NME for Train: %.2f %%  m = %d\n', mean(nme_train)*100 , m );
    
        % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
        pTrue_test = zeros( length(map_test{m}), 2*NUM_LANDMARKS);
        pInit_test = zeros( length(map_test{m}), 2*NUM_LANDMARKS);
        for i = 1 : length(map_test{m})
            item = arr_imgs_attri_test( map_test{m}(i) );
            true_shape = item.true_coord_all_landmarks;
            true_shape_row = [true_shape(1,:) true_shape(2,:)];
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
                MTCNN_keypoints = item.coord_just_keypoints;
                x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
                y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            elseif strcmp( train_para.box_type,'crop_true_keypoints' )
                true_keypoints = item.true_coord_all_landmarks;
                assert( NUM_LANDMARKS == 29 );
                true_keypoints = true_keypoints(:,[17 18 21 23 24] );
                x_min = min(true_keypoints(1,:)); x_max = max(true_keypoints(1,:));
                y_min = min(true_keypoints(2,:)); y_max = max(true_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            else
                error('remain to be developed.');
            end

            pTrue_test(i,:) = true_shape_row;
            init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
            arr_imgs_attri_test( map_test{m}(i) ).init_shapes.(train_para.shape_initialization) = init_shapes;
            pInit_test(i,:) = init_shapes;
    %         % debugging : display
    %         close all; figure(i);
    %         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
    %         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
    % 
    %         de = 0;
        end
        % Report NME
        nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
        fprintf('Report NME for Test: %.2f %%  m = %d\n', mean(nme_test)*100 , m );
    end
    
    % % % Save to file
    arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n', Normalization_file );
        
    arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n', Normalization_file );
end

%% mean_shape_with_box_augmentation_official
if strcmp( train_para.shape_initialization , 'RCPR_box_aug_rand_shapes' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
       
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        w = size(item.img,2); h = size(item.img,1);
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        %% Randomly select L shapes from training set
        candidate = [ 1:i-1 i+1:length(arr_imgs_attri) ];
        select = datasample( candidate, L, 'replace', false );
        init_shapes = zeros(L, NUM_LANDMARKS*2);
        for j = 1 : L
            shape_wrt_box = shapes_wrt_box( select(j), : );
            init_shapes(j,:) = shapeGt('reprojectPose', model, shape_wrt_box, box );
        end
        % make effect
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
        % Calculate NME
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*L +1): (i*L),:) = repmat( true_shape_row, L,1);
        pInit( ((i-1)*L+1) : (i*L),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    train_N = length( arr_imgs_attri );
    save( Normalization_file, 'arr_imgs_attri','-v7.3');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end

        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );

        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test( i,:) = init_shapes;
        % Calculate NME
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue_test( i,:) = true_shape_row;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end

if strcmp( train_para.shape_initialization , 'MTCNN_box_rand_shapes' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
       
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        w = size(item.img,2); h = size(item.img,1);
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        %% Randomly select L shapes from training set
        candidate = [ 1:i-1 i+1:length(arr_imgs_attri) ];
        select = datasample( candidate, L, 'replace', false );
        init_shapes = zeros(L, NUM_LANDMARKS*2);
        for j = 1 : L
            shape_wrt_box = shapes_wrt_box( select(j), : );
            init_shapes(j,:) = shapeGt('reprojectPose', model, shape_wrt_box, box );
        end
        % make effect
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
        % Calculate NME
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*L +1): (i*L),:) = repmat( true_shape_row, L,1);
        pInit( ((i-1)*L+1) : (i*L),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    train_N = length( arr_imgs_attri );
    save( Normalization_file, 'arr_imgs_attri','-v7.3');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end

        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );

        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test( i,:) = init_shapes;
        % Calculate NME
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue_test( i,:) = true_shape_row;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end

if strcmp( train_para.shape_initialization , 'Official_box_rand_shapes' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
       
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        w = size(item.img,2); h = size(item.img,1);
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        %% Randomly select L shapes from training set
        candidate = [ 1:i-1 i+1:length(arr_imgs_attri) ];
        select = datasample( candidate, L, 'replace', false );
        init_shapes = zeros(L, NUM_LANDMARKS*2);
        for j = 1 : L
            shape_wrt_box = shapes_wrt_box( select(j), : );
            init_shapes(j,:) = shapeGt('reprojectPose', model, shape_wrt_box, box );
        end
        % make effect
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
        % Calculate NME
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*L +1): (i*L),:) = repmat( true_shape_row, L,1);
        pInit( ((i-1)*L+1) : (i*L),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    train_N = length( arr_imgs_attri );
    save( Normalization_file, 'arr_imgs_attri','-v7.3');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end

        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );

        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test( i,:) = init_shapes;
        % Calculate NME
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue_test( i,:) = true_shape_row;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end

%% mean_shape_with_box_augmentation_official
if strcmp( train_para.shape_initialization , 'RCPR_box_aug_rand_shapes_poses' )
    if ~exist('arr_imgs_attri_train')
        Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
        load( Normalization_file );
        arr_imgs_attri_train = arr_imgs_attri;
    end
    if ~exist('arr_imgs_attri_test')
        Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
        load( Normalization_file );
        arr_imgs_attri_test = arr_imgs_attri;
    end
    L = train_para.L_num_initial_shapes;
    
    %%% Note: Divide three poses !
    map_pose{1}=[]; map_pose{2}=[]; map_pose{3}=[];
    map_test_pose{1}=[]; map_test_pose{2}=[]; map_test_pose{3}=[];
    for i = 1 : length(arr_imgs_attri_train)
        item = arr_imgs_attri_train(i);
        if item.pose_angles(1) < -5
            map_pose{1} = [ map_pose{1} i ];
        end
        if item.pose_angles(1) > -15 & item.pose_angles(1) < 15
            map_pose{2} = [ map_pose{2} i ];
        end
        if item.pose_angles(1) > 5
            map_pose{3} = [ map_pose{3} i ];
        end
    end
    map_test_pose{1}=[];map_test_pose{2}=[];map_test_pose{3}=[];
    for i = 1 : length(arr_imgs_attri_test)
        item = arr_imgs_attri_test(i);
        if item.pose_angles(1) < -10
            map_test_pose{1} = [ map_test_pose{1} i ];
        elseif item.pose_angles(1) >= -10 & item.pose_angles(1) <= 10
            map_test_pose{2} = [ map_test_pose{2} i ];
        else
            map_test_pose{3} = [ map_test_pose{3} i ];
        end
    end
    save( ['map_poses_',DATA_SET,'.mat'],'map_pose' );
    save( ['map_test_poses_',DATA_SET,'.mat'],'map_test_pose' );
    assert( length(map_pose{1})>0 & length(map_pose{2})>0  & length(map_pose{3})>0  );
    assert( length(map_test_pose{1})>0 & length(map_test_pose{2})>0 & length(map_test_pose{3})>0  );

    for m = 1 : 3
        
        % % % Get shapes w.r.t MTCNN box of all training samples
        shapes_wrt_box = zeros( length(map_pose{m}), 2*NUM_LANDMARKS );
        for i = 1 : length( map_pose{m} )
            item = arr_imgs_attri_train( map_pose{m}(i) );
            true_coords = item.true_coord_all_landmarks;
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
                MTCNN_keypoints = item.coord_just_keypoints;
                x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
                y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            elseif strcmp( train_para.box_type,'crop_true_keypoints' )
                true_keypoints = item.true_coord_all_landmarks;
                assert( NUM_LANDMARKS == 29 );
                true_keypoints = true_keypoints(:,[17 18 21 23 24] );
                x_min = min(true_keypoints(1,:)); x_max = max(true_keypoints(1,:));
                y_min = min(true_keypoints(2,:)); y_max = max(true_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            else
                error('remain to be developed.');
            end
            norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
            norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
            shapes_wrt_box(i,:) = norm_shape;
        end
        % Build the multivariate normal distribution model for
        % shapes_wrt_box
        mean_shapes_wrt_box = mean( shapes_wrt_box );
       
        % Generate initial shapes for each training sample( raw image
        % coords )
        pTrue = zeros( length(map_pose{m})*L, 2*NUM_LANDMARKS);
        pInit = zeros( length(map_pose{m})*L, 2*NUM_LANDMARKS);
        for i = 1 : length(map_pose{m})
            item = arr_imgs_attri_train( map_pose{m}(i) );
            w = size(item.img,2); h = size(item.img,1);
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
                MTCNN_keypoints = item.coord_just_keypoints;
                x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
                y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            elseif strcmp( train_para.box_type,'crop_true_keypoints' )
                true_keypoints = item.true_coord_all_landmarks;
                assert( NUM_LANDMARKS == 29 );
                true_keypoints = true_keypoints(:,[17 18 21 23 24] );
                x_min = min(true_keypoints(1,:)); x_max = max(true_keypoints(1,:));
                y_min = min(true_keypoints(2,:)); y_max = max(true_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            else
                error('remain to be developed.');
            end
            %% Select initial shapes
            candidate = map_pose{m};
            candidate(i) = [];
            select = datasample( [1:i-1 i+1:length(map_pose{m})], L, 'replace', false );
            init_shapes = zeros(L, NUM_LANDMARKS*2);
            for j = 1 : L
                init_shapes(j,:) = shapeGt('reprojectPose', model, shapes_wrt_box( select(j),:), box );
            end   
            arr_imgs_attri_train( map_pose{m}(i) ).init_shapes.(train_para.shape_initialization) = init_shapes;

            true_shape = item.true_coord_all_landmarks;
            true_shape_row = [true_shape(1,:) true_shape(2,:)];
            pTrue( ((i-1)*L+1): (i*L),:) = repmat( true_shape_row, L,1);

            
            pInit( ((i-1)*L+1): (i*L),:) = init_shapes;
                % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(item.img, item.true_coord_all_landmarks,  struct('box',box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(item.img, init_shapes(1,:),               struct('box',box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(item.img, init_shapes(L,:),               struct('box',box));imshow(tmp);
%             de = 0;
        end
        % Report NME
        nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
        fprintf('Report NME for Train: %.2f %%  m = %d\n', mean(nme_train)*100 , m );
    
        % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
        pTrue_test = zeros( length(map_test_pose{m}), 2*NUM_LANDMARKS);
        pInit_test = zeros( length(map_test_pose{m}), 2*NUM_LANDMARKS);
        for i = 1 : length(map_test_pose{m})
            item = arr_imgs_attri_test( map_test_pose{m}(i) );
            true_shape = item.true_coord_all_landmarks;
            true_shape_row = [true_shape(1,:) true_shape(2,:)];
            if strcmp( train_para.box_type , 'MTCNN' )
                box = item.MTCNN_box;
            elseif strcmp( train_para.box_type , 'official' )
                box = item.official_box;
            elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
                MTCNN_keypoints = item.coord_just_keypoints;
                x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
                y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            elseif strcmp( train_para.box_type,'crop_true_keypoints' )
                true_keypoints = item.true_coord_all_landmarks;
                assert( NUM_LANDMARKS == 29 );
                true_keypoints = true_keypoints(:,[17 18 21 23 24] );
                x_min = min(true_keypoints(1,:)); x_max = max(true_keypoints(1,:));
                y_min = min(true_keypoints(2,:)); y_max = max(true_keypoints(2,:));
                box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
            else
                error('remain to be developed.');
            end

            pTrue_test(i,:) = true_shape_row;
            init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );
            arr_imgs_attri_test( map_test_pose{m}(i) ).init_shapes.(train_para.shape_initialization) = init_shapes;
            pInit_test(i,:) = init_shapes;
    %         % debugging : display
%             close all; figure(i);
%             subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%     
%             de = 0;
        end
        % Report NME
        nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
        fprintf('Report NME for Test: %.2f %%  m = %d\n', mean(nme_test)*100 , m );
    end
    
    % % % Save to file
    arr_imgs_attri = arr_imgs_attri_train;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n', Normalization_file );
        
    arr_imgs_attri = arr_imgs_attri_test;
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n', Normalization_file );
end

if strcmp( train_para.shape_initialization , 'crop_MTCNN_points_mean_shape' )
    warning('Currently, for marker = Synthetical only !\n');
    marker = 'Synthetical';
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];

    load( Normalization_file );

    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
    samples = repmat( mean_shapes_wrt_box, length(arr_imgs_attri)*train_para.L_num_initial_shapes, 1 );
       
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        w = size(item.img,2); h = size(item.img,1);
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        %% Randomly select L shapes from training set
        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );

        % make effect
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        
        % Calculate NME
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*L +1): (i*L),:) = repmat( true_shape_row, L,1);
        pInit( ((i-1)*L+1) : (i*L),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    train_N = length( arr_imgs_attri );
    save( Normalization_file, 'arr_imgs_attri', '-v7.3');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ 'Original', '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri)*L, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
         
        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );

        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test( ((i-1)*L +1): (i*L),:) = init_shapes;
        % Calculate NME
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue_test( ((i-1)*L +1): (i*L),:) = repmat( true_shape_row, L,1);
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end

%% mean_shape_with_box_augmentation_official
if strcmp( train_para.shape_initialization , 'multivariate_normal_sampling' )
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    load( Normalization_file );
    N = length(arr_imgs_attri);
    L = train_para.L_num_initial_shapes;

    % Get shapes w.r.t MTCNN box of all training samples
    shapes_wrt_box = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS );
    for i = 1 : length( arr_imgs_attri )
        true_coords = arr_imgs_attri(i).true_coord_all_landmarks;
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        norm_shape(1:NUM_LANDMARKS)                 = double(( true_coords(1,:)-(box(1)+box(3)/2) )) / double( box(3)/2 );
        norm_shape(NUM_LANDMARKS+1:2*NUM_LANDMARKS) = double(( true_coords(2,:)-(box(2)+box(4)/2) )) / double( box(4)/2 );
        shapes_wrt_box(i,:) = norm_shape;
    end
    % Build the multivariate normal distribution model for
    % shapes_wrt_box
    mean_shapes_wrt_box = mean( shapes_wrt_box );
    cov_shapes_wrt_box  = cov(shapes_wrt_box);
    samples = mvnrnd( mean_shapes_wrt_box, cov_shapes_wrt_box, length(arr_imgs_attri)*train_para.L_num_initial_shapes );
    % Generate initial shapes for each training sample( raw image
    % coords )
    pTrue = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    pInit = zeros( length(arr_imgs_attri)*train_para.L_num_initial_shapes, 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = repmat( true_shape_row, train_para.L_num_initial_shapes,1);
        init_shapes = shapeGt('reprojectPose', model, samples( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:), repmat( box, train_para.L_num_initial_shapes,1) );
        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit( ((i-1)*train_para.L_num_initial_shapes+1): (i*train_para.L_num_initial_shapes),:) = init_shapes;
%             % debugging : display
%             close all; figure(i);
%             subplot(1,3,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             subplot(1,3,3); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(L,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%             de = 0;
    end
    % Save the result
    nme_train = mean( shapeGt( 'dist',model, pInit, pTrue ) );
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TRAIN', '_normalization.mat' ];
    train_N = length( arr_imgs_attri );
    save( Normalization_file, 'arr_imgs_attri','-v7.3');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Train: %.2f %%\n', mean(nme_train)*100 );
    
    % %%%%%%%%%%%% Generate initial shape for test samples %%%%%%%%%%%%
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    load( Normalization_file );
    pTrue_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    pInit_test = zeros( length(arr_imgs_attri), 2*NUM_LANDMARKS);
    for i = 1 : length(arr_imgs_attri)
        item = arr_imgs_attri(i);
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        if strcmp( train_para.box_type , 'MTCNN' )
            box = arr_imgs_attri(i).MTCNN_box;
        elseif strcmp( train_para.box_type , 'official' )
            box = arr_imgs_attri(i).official_box;
        elseif strcmp( train_para.box_type,'crop_MTCNN_keypoints' )
            MTCNN_keypoints = arr_imgs_attri(i).coord_just_keypoints;
            x_min = min(MTCNN_keypoints(1,:)); x_max = max(MTCNN_keypoints(1,:));
            y_min = min(MTCNN_keypoints(2,:)); y_max = max(MTCNN_keypoints(2,:));
            box = [ x_min, y_min, x_max-x_min, y_max-y_min ];
        else
            error('remain to be developed.');
        end

        init_shapes = shapeGt('reprojectPose', model, mean_shapes_wrt_box, box );

        arr_imgs_attri(i).init_shapes.(train_para.shape_initialization) = init_shapes;
        pInit_test( i,:) = init_shapes;
        % Calculate NME
        true_shape = arr_imgs_attri(i).true_coord_all_landmarks;
        true_shape_row = [true_shape(1,:) true_shape(2,:)];
        pTrue_test( i,:) = true_shape_row;
%         % debugging : display
%         close all; figure(i);
%         subplot(1,2,1); tmp=disp_landmarks(arr_imgs_attri(i).img, arr_imgs_attri(i).true_coord_all_landmarks,  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
%         subplot(1,2,2); tmp=disp_landmarks(arr_imgs_attri(i).img, init_shapes(1,:),  struct('box',arr_imgs_attri(i).MTCNN_box));imshow(tmp);
% 
%         de = 0;
    end
    nme_test = mean( shapeGt( 'dist',model, pInit_test, pTrue_test ) );
    % Save the result
    Normalization_file = [ marker, '_', DATA_SET, '_', 'TEST', '_normalization.mat' ];
    save( Normalization_file, 'arr_imgs_attri');
    fprintf('%s saved\n',Normalization_file );
    fprintf('Report NME for Test: %.2f %%\n', mean(nme_test)*100 );
end