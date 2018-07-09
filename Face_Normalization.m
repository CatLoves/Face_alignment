%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Face Normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Key points:
% % 1. Face detection
% % 2. Five landmarks localization
% % 3. Rotation
% % 4. Scaling
%% Environmental variables
clear arr_imgs_attri;
global Input_images_dir;
global TRAIN_OR_TEST;
global NUM_LANDMARKS;
global imageNum;            % num of images
global arr_imgs_attri;
global NORMALIZED_DIR;
global LOGS_DIR;
global ROLL_THRESHOLD;
global N;
global model;
global imgs_attri_train;
global imgs_attri_test;
global GLOBAL_IOD;
global PREPROCESSING_DIR;
global CAFFE_HOME;
global DATA_SET;
global PREPROCESSING_RESULT_FILE;
global train_para;
global feat_para;
global mean_shapes;
global DICTIONARY_RESULT_FILE;
global ERROR_DIR;
global DATA_SET_DIR;
global marker;
cd( PREPROCESSING_DIR );
%% If Normalization_file already exists, just skip this file
cd( PREPROCESSING_DIR );
Normalization_file = [ marker, '_', DATA_SET, '_', TRAIN_OR_TEST, '_normalization.mat' ];
will_skip = 0;
if exist( Normalization_file )
    will_skip = 1;
    fprintf('Normalization file already exist. \n Will skip the face normalization process in 3 seconds ...\n');
    pause(1);
    fprintf('Will skip the face normalization process in 2 seconds ...\n');
    pause(1);
    fprintf('Will skip the face normalization process in 1 seconds ...\n');
    pause(1);
end
de = 0;

if will_skip == 0

    %%%%%%%%%%%%%%%%%%% Attributes of every image. %%%%%%%%%%%%%%%%%%%
    tmp_arr_imgs_attri(1)       = struct ...
    ( 'image_fullname' , [],...                                 % full image name
      'true_coord_all_landmarks',  [],...                       % ground_truth coordinates for all landmarks, raw image coords( no rotation & scaling )
      'coord_just_keypoints',      [],...                       % just few points from DCNN, in the order:left_eye , right_eye , nose , left_mouth , right_mouth, 
      'estimated_coord_all_landmarks',   [],...                 % estimated coordinates of all landmarks
      'face_box' , [] ,  ...                                    % face bounding official_box , various kinds, MTCNN/Viola_Jones/Customed, NOTE: Based on normalized image rather than original image
      'delimiter' , [] , ...                                    % No meaning , to separate fields
      'norm_img' , [] , ...                                     % Normalized image
      'norm_true_coord_all_landmarks', [],...                   % normalized
      'norm_coord_just_keypoints', [],...                       % normalized coords of keypoints
      'norm_estimated_coord_all_landmarks',   [],...            % estimated coordinates of all landmarks
      'norm_regressed_init_phi' , [] , ...                      % Initial shape, ( regression via image features )
      'true_shape_relative_to_box' ,[] ,...                     % For random initialization or pose based or something else
      'delimiter2' , [] , ...                                   % Delimiters , no meaning
      'pose_angles',			   [],...                       % in the order: yaw , pitch , roll 
      'extra_info' , [] ...                                     % for example, init_scale_ratio , delta_x , delta_y , etc
    );
    arr_imgs_attri = tmp_arr_imgs_attri;
    clear tmp_arr_imgs_attri;

    %% %%%%%%%%%%%%%%%%% Load MT_CNN model %%%%%%%%%%%%%%%%%%
    % use cpu
    caffe.set_mode_cpu();
    % path of model files
    caffe_model_path = strcat( CAFFE_HOME , '/MTCNNv2/model' );
    threshold = [0.5 0.5 0.3];
    % scale factor
    factor=0.79;
    % minsize: must be set latter
    % load caffe models
    prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
    model_dir = strcat(caffe_model_path,'/det1.caffemodel');
    PNet=caffe.Net(prototxt_dir,model_dir,'test');
    prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
    model_dir = strcat(caffe_model_path,'/det2.caffemodel');
    RNet=caffe.Net(prototxt_dir,model_dir,'test');	
    prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
    model_dir = strcat(caffe_model_path,'/det3.caffemodel');
    ONet=caffe.Net(prototxt_dir,model_dir,'test');
    prototxt_dir =  strcat(caffe_model_path,'/det4.prototxt');
    model_dir =  strcat(caffe_model_path,'/det4.caffemodel');
    LNet=caffe.Net(prototxt_dir,model_dir,'test');
    % set directory
    cd( strcat( CAFFE_HOME , '/MTCNNv2' ) );

    % Store IOD based distances
    IOD_dis_cnt = 0;
    no_face_cnt = 0;        % how many faces are NOT DETECTED by MTCNN
    clear 'IOD_dis_stat';

    %% Main loop
    %%%%%%%%%%%%%%%%%%%%%  %%%%%%%%%%%%%%%%%%%%%
    imagelist_file = [ marker, '_Input_imagelist.txt' ];
    f_in = fopen( imagelist_file , 'r'); assert( f_in ~= -1 );
    current_file = fgetl( f_in ); cur_index = 0;

    while ischar( current_file )
        %% Read the image
        cur_index = cur_index + 1;
        % Load the current file
        img = imread( current_file );
        % check channels, make sure it is 3 channel
        if size( img , 3 ) == 1
            img = repmat( img , 1 , 1 , 3 );
        end

        % store image_name
        arr_imgs_attri(cur_index).image_fullname = current_file;

        %% %%%%%%%% read ground_truth coordinates & face bounding boxes( COFW: mask & occlu_label )%%%%%%%%%%
        [path, name, ext] = fileparts( current_file );
        coord_file = [ path, '/', name, '.pts' ];
        box_file = [ path, '/', name, '_box.pts' ];
        arr_imgs_attri(cur_index).raw_true_coord_all_landmarks = shapeGt('read_coords', coord_file, NUM_LANDMARKS );
        arr_imgs_attri(cur_index).raw_official_box             = shapeGt('read_box', box_file );
        % Read the occlusion mask ( If DATA_SET='COFW' )
        if strcmp( DATA_SET, 'COFW' ) || strcmp( DATA_SET, 'cofw' )
            mask_file = [path, '/', name, '_mask.mat'];
            if exist( mask_file,'file' )
                mask = load( mask_file, 'mask' );
                mask = mask.mask;
                assert( size(img,1)==size(mask,1) & size(img,2)==size(mask,2) );
            else
                mask = false( size(img,1), size(img,2) );
                fprintf('%s not found.\n',mask_file );
            end
            arr_imgs_attri(cur_index).raw_mask = mask;
        end
        % Read the occlu_label ( If DATA_SET='COFW' )
        if strcmp( DATA_SET, 'COFW' ) || strcmp( DATA_SET, 'cofw' )
            occlu_label = [path, '/', name, '_occlu_label.mat'];
            load( occlu_label);
            if strcmp( TRAIN_OR_TEST, 'TRAIN' )
                arr_imgs_attri(cur_index).occlu_label = occlu_label;
            elseif strcmp( TRAIN_OR_TEST, 'TEST' )
                arr_imgs_attri(cur_index).occlu_label = occlusion_label;
            else error('invalid'); end
        end

        %% %%%%%%%%%% MT_CNN locates face boxes and five points  %%%%%%%%%%%%
        minl    = min( [size(img,1) size(img,2)]);
        if NUM_LANDMARKS == 29
            %minsize = fix( minl * 0.03 );    % for faster detection
            minsize = 10;
        elseif NUM_LANDMARKS == 68
            minsize = fix( minl * 0.05 );
        else error('invalid value'); 
        end
        [ boudingboxes , points ]= detect_face( img, minsize , PNet , RNet , ONet , LNet , threshold , false , factor );
        % If no face is detected, can skip
        if size( boudingboxes, 1 ) == 0
            fprintf('No face is detected for %s  Skipping...\n', current_file );
            cur_index = cur_index - 1;
            current_file = fgetl( f_in );
            continue;
        end

        % % Find the right face box, i.e., closest to the official box
        min_dis = 100000;
        min_ind = -1;
        for i = 1 : size(boudingboxes,1)
            cur_box = [ boudingboxes(i,1:2) boudingboxes(i,3:4)-boudingboxes(i,1:2) ];
            cur_dis = mean( abs( arr_imgs_attri(cur_index).raw_official_box - cur_box ) );
            if cur_dis < min_dis
                min_dis = cur_dis;
                min_ind = i;
            end
        end
        assert( min_ind ~= -1 );
        % % Check whether the detected facial box is reasonable by computing overlap
        raw_MTCNN_box = [ boudingboxes(min_ind,1:2) boudingboxes(min_ind,3:4)-boudingboxes(min_ind,1:2) ];
        overlap_ratio = shapeGt('calcOverlapRatio', arr_imgs_attri(cur_index).raw_official_box, raw_MTCNN_box );
        % Skip the image if overlap_ratio < 0.5
        if overlap_ratio < 0.5
            fprintf('MTCNN detection failed, skipping...\n');
            cur_index = cur_index - 1;
            current_file = fgetl( f_in );
            % visualization
            close all; figure(1);
            img = imread(arr_imgs_attri(cur_index).image_fullname);
            true_coords = arr_imgs_attri(cur_index).raw_true_coord_all_landmarks;
            official_box = arr_imgs_attri(cur_index).raw_official_box;
            mtcnn_box = raw_MTCNN_box;
            subplot(1,2,1); tmp=disp_landmarks(img, true_coords, struct('box',official_box) ); imshow(tmp); title('offcial box');
            subplot(1,2,2); tmp=disp_landmarks(img, true_coords, struct('box',mtcnn_box) ); imshow(tmp); title('offcial box');
            de = 0;
            continue;
        end
        arr_imgs_attri(cur_index).raw_MTCNN_box = raw_MTCNN_box;

        % % Find the Five landmarks( eyes, nose tip, mouth corners )
        arr_imgs_attri(cur_index).raw_MTCNN_five_landmarks = reshape( points(:,min_ind), 5, 2 )';
        % Check the validity
        keypoints = arr_imgs_attri(cur_index).raw_MTCNN_five_landmarks;
        min_x = min( keypoints(1,:) );
        max_x = max( keypoints(1,:) );
        min_y = min( keypoints(2,:) );
        max_y = max( keypoints(2,:) );
        
        if min_x < 0 || max_x > size(img,2) || min_y < 0 || max_y > size(img,1) || norm(keypoints(:,1)-keypoints(:,2)) < 20 % IOD should not be less than 20 pixels
            % Handle the exception
            fprintf('Keypoints out of range! MTCNN detection failed, skipping...\n');
            cur_index = cur_index - 1;
            current_file = fgetl( f_in );
        end

        %% Scaling the face image & true_coord_all_landmarks & MTCNN_five_landmarks , so that IOD = GLOBAL_IOD( 100 pixels )
        cur_IOD = norm( arr_imgs_attri(cur_index).raw_MTCNN_five_landmarks(:,1) - arr_imgs_attri(cur_index).raw_MTCNN_five_landmarks(:,2) );
        % error detection
        if cur_IOD < 10 || cur_IOD > 1500;error('IOD Exception occurs! please note the img: %s \n', current_file );end

        arr_imgs_attri( cur_index ).extra_info.init_scaling_ratio = GLOBAL_IOD / cur_IOD;
        arr_imgs_attri(cur_index).img                      = imresize( img , arr_imgs_attri( cur_index ).extra_info.init_scaling_ratio );
        if strcmp( DATA_SET, 'COFW' ) || strcmp( DATA_SET, 'cofw' ); arr_imgs_attri(cur_index).mask = imresize( mask, arr_imgs_attri( cur_index ).extra_info.init_scaling_ratio );end
        arr_imgs_attri(cur_index).true_coord_all_landmarks = arr_imgs_attri(cur_index).raw_true_coord_all_landmarks * arr_imgs_attri( cur_index ).extra_info.init_scaling_ratio;
        arr_imgs_attri(cur_index).official_box             = arr_imgs_attri(cur_index).raw_official_box             * arr_imgs_attri( cur_index ).extra_info.init_scaling_ratio;
        % MTCNN 
        arr_imgs_attri(cur_index).MTCNN_box                = arr_imgs_attri(cur_index).raw_MTCNN_box                * arr_imgs_attri( cur_index ).extra_info.init_scaling_ratio;
        arr_imgs_attri(cur_index).MTCNN_five_landmarks     = arr_imgs_attri(cur_index).raw_MTCNN_five_landmarks     * arr_imgs_attri( cur_index ).extra_info.init_scaling_ratio;
        arr_imgs_attri(cur_index).coord_just_keypoints     = arr_imgs_attri(cur_index).MTCNN_five_landmarks;

        if isfield( arr_imgs_attri(cur_index), 'gt_box' )
            arr_imgs_attri(cur_index).gt_box = gt_box * arr_imgs_attri( cur_index ).extra_info.init_scaling_ratio;
        end
    %     % debugging
    %     close all; figure(cur_index);
    %     subplot(1,2,1);title('official box'); tmp=disp_landmarks(arr_imgs_attri(cur_index).img, arr_imgs_attri(cur_index).coord_just_keypoints, struct('box',arr_imgs_attri(cur_index).official_box) );imshow(tmp);
    %     subplot(1,2,2);title('true'); tmp=disp_landmarks(arr_imgs_attri(cur_index).img, arr_imgs_attri(cur_index).coord_just_keypoints, struct('box',arr_imgs_attri(cur_index).MTCNN_box) );imshow(tmp);
    %     de = 0;

        %% Rotate the face image, based on head pose estimated by MTCNN five landmarks
        [yaw, pitch, roll] = estimateHeadPose( arr_imgs_attri(cur_index).coord_just_keypoints );
        arr_imgs_attri(cur_index).pose_angles = [ yaw pitch roll ];
        % Rotate the image around the mean(keypoints)
        Rotation_center                = mean( arr_imgs_attri(cur_index).coord_just_keypoints, 2 );
        rs                             = my_rotate_img( arr_imgs_attri(cur_index).img, Rotation_center, roll );
        arr_imgs_attri(cur_index).img  = rs.rotated_img;
        % if DATA_SET = COFW
        if isfield( arr_imgs_attri(cur_index), 'mask' )
            rs_mask                                  = my_rotate_img( arr_imgs_attri(cur_index).mask , Rotation_center , roll );
            arr_imgs_attri(cur_index).mask           = rs_mask.rotated_img;
        end
        ori_true_coords = arr_imgs_attri(cur_index).true_coord_all_landmarks;
        arr_imgs_attri(cur_index).true_coord_all_landmarks  = my_rotate_coords( arr_imgs_attri(cur_index).true_coord_all_landmarks , roll , rs.params );
        arr_imgs_attri(cur_index).coord_just_keypoints      = my_rotate_coords( arr_imgs_attri(cur_index).coord_just_keypoints     , roll , rs.params );
        % For inverse transformation of coords
        arr_imgs_attri( cur_index ).extra_info.roll = roll;
        arr_imgs_attri( cur_index ).extra_info.params = rs.params;
        % % % % % Adjust box: Rotate the four corners of the box, then compute the
        % new box
        img_width  = size( arr_imgs_attri(cur_index).img, 2 );
        img_height = size( arr_imgs_attri(cur_index).img, 1 );
        x_o = arr_imgs_attri(cur_index).official_box(1); % x_coord of left upper vertex of official box
        y_o = arr_imgs_attri(cur_index).official_box(2); % y_coord of left upper vertex
        w_o = arr_imgs_attri(cur_index).official_box(3); % width
        h_o = arr_imgs_attri(cur_index).official_box(4); % height
        corners_o = [ x_o x_o+w_o x_o x_o+w_o; y_o y_o y_o+h_o y_o+h_o ];
        rotated_corners_o = my_rotate_coords( corners_o, roll , rs.params );
        % Crop the rectangle
        left  = max( min(rotated_corners_o(1,:)), 1 );
        right = min( max(rotated_corners_o(1,:)), img_width );
        up    = max( min(rotated_corners_o(2,:)), 1 );
        down  = min( max(rotated_corners_o(2,:)), img_height );
        arr_imgs_attri(cur_index).official_box = [ left, up, right-left, down-up ];
        
        % For the MTCNN box
        x_mtcnn = arr_imgs_attri(cur_index).MTCNN_box(1); % x_coord of left upper vertex of official box
        y_mtcnn = arr_imgs_attri(cur_index).MTCNN_box(2); % y_coord of left upper vertex
        w_mtcnn = arr_imgs_attri(cur_index).MTCNN_box(3); % width
        h_mtcnn = arr_imgs_attri(cur_index).MTCNN_box(4); % height
        corners_mtcnn = [ x_mtcnn x_mtcnn+w_mtcnn x_mtcnn x_mtcnn+w_mtcnn; y_mtcnn y_mtcnn y_mtcnn+h_mtcnn y_mtcnn+h_mtcnn ];
        rotated_corners_mtcnn = my_rotate_coords( corners_mtcnn, roll , rs.params );
        % Crop the rectangle
        left  = max( min(rotated_corners_mtcnn(1,:)), 1 );
        right = min( max(rotated_corners_mtcnn(1,:)), img_width );
        up    = max( min(rotated_corners_mtcnn(2,:)), 1 );
        down  = min( max(rotated_corners_mtcnn(2,:)), img_height );
        arr_imgs_attri(cur_index).MTCNN_box = [ left, up, right-left, down-up ];
        
%         % debugging: inverse transform to original
%         ori = [ ori_true_coords(1,:) ori_true_coords(2,:) ];
%         recovered = my_rotate_coords_back( arr_imgs_attri(cur_index).true_coord_all_landmarks, roll, rs.params );
%         recovered = [recovered(1,:) recovered(2,:) ];
%         dis = shapeGt('dist', model, recovered,  ori );

%       %% Crop the face image, the only purpose is to save memory usage
        padding = 100;
        w = size( arr_imgs_attri(cur_index).img, 2 ); h = size( arr_imgs_attri(cur_index).img,  1);
        left  = round( max( min( arr_imgs_attri(cur_index).coord_just_keypoints(1,:) ) - 1.3*padding, 1 ) );
        right = round( min( max( arr_imgs_attri(cur_index).coord_just_keypoints(1,:) ) + 1.3*padding, w ) );
        upper = round( max( min( arr_imgs_attri(cur_index).coord_just_keypoints(2,:) ) - padding, 1 ) );
        down  = round( min( max( arr_imgs_attri(cur_index).coord_just_keypoints(2,:) ) + 2 * padding, h ) );

        % Adjust coords
        arr_imgs_attri(cur_index).img  = arr_imgs_attri(cur_index).img( upper:down, left:right, :);
        if strcmp(DATA_SET, 'COFW') || strcmp( DATA_SET, 'cofw' )
            arr_imgs_attri(cur_index).mask = arr_imgs_attri(cur_index).mask( upper:down, left:right, :);
        end
        w = size(arr_imgs_attri(cur_index).img,2);       h = size(arr_imgs_attri(cur_index).img,1);
        arr_imgs_attri(cur_index).true_coord_all_landmarks = arr_imgs_attri(cur_index).true_coord_all_landmarks - repmat( [left-1,upper-1]',1,NUM_LANDMARKS);
        arr_imgs_attri(cur_index).coord_just_keypoints     = arr_imgs_attri(cur_index).coord_just_keypoints - repmat( [left-1,upper-1]',1,5);
        arr_imgs_attri(cur_index).MTCNN_five_landmarks     = arr_imgs_attri(cur_index).MTCNN_five_landmarks - repmat( [left-1,upper-1]',1,5);
        arr_imgs_attri(cur_index).official_box(1)          = arr_imgs_attri(cur_index).official_box(1) - (left-1);
        arr_imgs_attri(cur_index).official_box(2)          = arr_imgs_attri(cur_index).official_box(2) - (upper-1);
        arr_imgs_attri(cur_index).official_box(1)          = max( arr_imgs_attri(cur_index).official_box(1), 1 );
        arr_imgs_attri(cur_index).official_box(2)          = max( arr_imgs_attri(cur_index).official_box(2), 1 );
        arr_imgs_attri(cur_index).official_box(3)          = min( arr_imgs_attri(cur_index).official_box(3), abs(w-arr_imgs_attri(cur_index).official_box(1)) );
        arr_imgs_attri(cur_index).official_box(4)          = min( arr_imgs_attri(cur_index).official_box(4), abs(h-arr_imgs_attri(cur_index).official_box(2)) );
        arr_imgs_attri(cur_index).MTCNN_box(1)             = arr_imgs_attri(cur_index).MTCNN_box(1) - (left-1);
        arr_imgs_attri(cur_index).MTCNN_box(2)             = arr_imgs_attri(cur_index).MTCNN_box(2) - (upper-1);
        arr_imgs_attri(cur_index).MTCNN_box(1)             = max( arr_imgs_attri(cur_index).MTCNN_box(1), 1 );
        arr_imgs_attri(cur_index).MTCNN_box(2)             = max( arr_imgs_attri(cur_index).MTCNN_box(2), 1 );
        arr_imgs_attri(cur_index).MTCNN_box(3)             = min( arr_imgs_attri(cur_index).MTCNN_box(3), abs(w-arr_imgs_attri(cur_index).MTCNN_box(1)) );
        arr_imgs_attri(cur_index).MTCNN_box(4)             = min( arr_imgs_attri(cur_index).MTCNN_box(4), abs(h-arr_imgs_attri(cur_index).MTCNN_box(2)) );
        arr_imgs_attri( cur_index ).extra_info.crop_position = [left-1,upper-1]';
        
        % transform back
        % step 1: shift
        trans = arr_imgs_attri(cur_index).true_coord_all_landmarks + repmat( arr_imgs_attri( cur_index ).extra_info.crop_position,1,NUM_LANDMARKS);
        % step 2: rotation
        trans =  my_rotate_coords_back( trans, arr_imgs_attri( cur_index ).extra_info.roll, arr_imgs_attri( cur_index ).extra_info.params );
        % step 3: scaling
        trans = trans ./ arr_imgs_attri( cur_index ).extra_info.init_scaling_ratio;
        % step 4: transform
        trans = [trans(1,:) trans(2,:) ];
        ori =[arr_imgs_attri(cur_index).raw_true_coord_all_landmarks(1,:) arr_imgs_attri(cur_index).raw_true_coord_all_landmarks(2,:) ];
        error(cur_index) = shapeGt('dist',model, trans, ori );
        de = 0;
    %     % debugging
%         close all; figure(cur_index);
%         subplot(1,4,1); tmp=disp_landmarks(arr_imgs_attri(cur_index).img,...
%                                                           arr_imgs_attri(cur_index).true_coord_all_landmarks,...
%                                                           struct('box',arr_imgs_attri(cur_index).MTCNN_box) );imshow(tmp);title('mtcnn box');
%         subplot(1,4,2); tmp=disp_landmarks(arr_imgs_attri(cur_index).img,...
%                                                           arr_imgs_attri(cur_index).true_coord_all_landmarks,...
%                                                           struct('box',arr_imgs_attri(cur_index).official_box) );imshow(tmp);title('official box');
%         subplot(1,4,3); imshow(arr_imgs_attri(cur_index).mask);title('occlu mask');
%         subplot(1,4,4); tmp=disp_landmarks(arr_imgs_attri(cur_index).img,arr_imgs_attri(cur_index).coord_just_keypoints); imshow(tmp);title('key points');
        de = 0;
%         subplot(1,3,2);title('mask');  imshow(arr_imgs_attri(cur_index).mask);
%         I = arr_imgs_attri(cur_index).img;
%         for i = 1 : size(arr_imgs_attri(cur_index).mask,1)
%             for j = 1 : size(arr_imgs_attri(cur_index).mask,2)
%                 if arr_imgs_attri(cur_index).mask(i,j) == true
%                     I(i,j,1) = 0; I(i,j,2) = 0; I(i,j,3) = 0;
%                 end
%             end
%         end
%         subplot(1,3,3); imshow(I); title('semi-mask');
% %     %     subplot(1,3,3);title('raw'); tmp=disp_landmarks( arr_imgs_attri(cur_index).img, arr_imgs_attri(cur_index).coord_just_keypoints, struct('box',arr_imgs_attri(cur_index).MTCNN_box) );imshow(tmp);
%          de = 0;

        %% Display progress and go to the next
        % Indicating progress
        fprintf( 'Processing progress:%d  \n' , cur_index  );
        % Go to the next
        current_file = fgetl( f_in );
    end  % cur_index = 1:imageNum
    % close the file, which is very important
    fclose( f_in );
    N = cur_index;

    %% Save  normalization results
    cd( PREPROCESSING_DIR );
    save( Normalization_file, 'arr_imgs_attri' , '-v7.3' );
    fprintf('********** Face Normalization Output: %s\n',Normalization_file);
    arr_imgs_attri = arr_imgs_attri(1:cur_index);
    
    %% Unit test 
    for i = round(linspace(1,length(arr_imgs_attri),10))
        item = arr_imgs_attri(i);
        name = item.image_fullname;
        [path,name,ext] = fileparts( name );
        img = item.img;
        coords = item.true_coord_all_landmarks;
        keypoints = item.coord_just_keypoints;
        mtcnn_box = item.MTCNN_box;
        % if the occlusion is considered
        if isfield( item, 'mask' )
            mask = item.mask;
            occlu_label = item.occlu_label;
        end

        close all; figure(i);
        subplot(1,4,1); tmp =disp_landmarks(img, coords,struct('box',mtcnn_box));imshow(tmp); title('true coords with mtcnn box');
        subplot(1,4,2); tmp =disp_landmarks(img, keypoints,struct('box',mtcnn_box));imshow(tmp); title('keypoints');

        % Display the mask, a litter tricky
        if isfield( item, 'mask' )
            semi_mask = img;
            for r = 1:size(img,1)
                for c = 1:size(img,2)
                    if mask(r,c)
                       semi_mask(r,c,1)=0; semi_mask(r,c,2)=0; semi_mask(r,c,3)=0;
                    end
                end
            end
            %figure(2);
            subplot(1,4,3); imshow(semi_mask); title('occlusion mask display');
            subplot(1,4,4); tmp =disp_landmarks(img, coords,struct('box',mtcnn_box,'occlu',arr_imgs_attri(i).occlu_label,'occlu_th',0.25));imshow(tmp); title('occlu label');
        end
        de = 0;
    end
end


