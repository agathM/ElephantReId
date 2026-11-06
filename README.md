#ElephantReId Pipeline
Definitions
By default body_part doesn't take into account left/right distinction for ears.

Data :
In the database folder :
- Video database with columns : video (filename of the video), time (creation time of the video), camera, mode (Night/Day), several_indivs(bool if several animals in the video), date, indiv (one line per individual that appears in the video), Description (of the individual location not entirely filled)
- Frame database with the same columns as the detection .csv plus the avg_conf (average confidence score over the track_id). All indiv are valid.
- Videos stored in the video folder, to be added in home_dir / video_dir
- To import new videos, add them to the video folder and actualize the video database and the frame database. In particular, the time and camera columns are needed for the tracking accross videos.
Models :
- 2 models are available : seg_ears_lr_v2.pt (Cyprien segmentation model with left_ear_seg and right_ear_seg classes)
                           head_ears_detect_09_08.pt (my head and ear detection model with classes : head_detect, left_ear_detect and right_ear_detect.
1) Detection

    Detections are run on every videos of the video_folder. Videos are sorted by camera and by time and detection is run with tracking. The tracker is reinitialized when the video comes from a new camera or when the creation time between two videos exceeds 5 minutes. For now the track id is not used. Some individuals have several track id in a single video for the segmented ear detections. I noted only one case of a single track id in the same video assigned to two different individuals.

   The detections are saved in crops_dir / model_filename / detection_class the detection class is {body_part}_{detect/seg} with distinction between left and right ear and the metadata in a .csv with the columns :
    - image (filename of the image),
   - video , frame (frame number), the coordinates x1,x2,y1,y2 of the bounding box for detect and the bounding box around the segmentation for seg needed to link to a detection to an individual with the frame database if there are several individuals on the video. Otherwise the detection is associated with the single individual appearing in the video through the video database.
   - conf (confidence score of the detection), track_id, area (width*height for a detection, number of pixels of the mask for segmentation), ratio (width/height)
   - indiv : indiv identified or nan
   
   Treated videos are saved in a pickle dictionnary. If all detections on the videos were labelized, they are never retreated except if you remove one or more from the set value of "all_detect". Videos with some detection not linked to an individual are save in "some_detect". You can choose to save them anyway by putting keep_detections_without_id to True in pipeline.yaml. You can also choose to save videos with detections if you want to check labelisation or to add new ones.


   Choose detections among ear_seg (Cyprien model), ear_detect, head_detect (my model), body_detect(MegaDetector)
   If you want to use a new detection model :
       If you have the .pt of a YOLO model (check conformity with yolo26n.pt) save it into the "models" folder and complete DETECT_PARAMS :
           Add the name of the detection {body_part}_{detect/seg} in the if conditions
           Put the name of the model file without extension as key of DETECT_PARAMS, and choose the confidence score
           If the model has two classes/body parts use same process a setdfault (first if).
           The detection classes must contain "detect" or "seg".

       If it is another detection model that needs to be imported, modify #MODEL_IMPORT in detect_and_track, defining the model, the detection classes to be kept and create your own process_video function with the same signatures as the others.

   
   You can modify the detection parameters in the pipeline.yaml : fps, tracker, identification_threshold (the minimal proportion of bounding box area
   of a segmented ear of the frame database to be contained in the detection bounding box to be linked to an individual).

   A label is assigned to each detection : {indiv_name}_{body_part}_{detect/seg}. If you want to take into account right/left distinction for ear in the label, modify the line (put ASSYMETRY to True in pipeline).
   
3) Filtering

    Once detections is performed a filtering step begins. For each detection, a confidence score, min_area, max_ratio and min_ratio can be choosen or put to null.
    A list of filter scripts can also be given for any detection. They must be stored in detection_filters. For now only seg_cut is available for ear_seg (cf           filtering description in the file).

   
4) Split

   A split is performed on every filtered detections. You can choose the event needed for the grouping of the videos, it can be day, week, month or a duration in minutes or hours (5min, 1h...). Then each videos are assigned to an event through their camera and time. All events with an individual that appears in a single event are assigned to the split set. If the total number of detection of a label is inferior to min_crop_per_label_for_val then all events with this individual go to train. Then for the max_run number than you specify in pipeline.yaml, events are shuffled randomly and processed one by one. If if an individual in the event is not in the training set then the event goes to the training set. If an individual is not in validation set but appears in only two events, the event goes to the validation set if force_val=True. Else a score is calculated if the event is assigned to training or validation. It is the proportion of detections assigned to the training set after event affectation (sum of differences between the number of detections per label in the training set on the total number of detections of this label) and the desired training ratio. The total number of detections of this label corresponds to the total number of detections, not the already assigned detections to training or validation.
   Are saved in split_files folder :
   A {split_version}_splits.csv contains run number, event and split column (train/val).
   A {split_version}_selected_images.csv contains same data as the detections csv except video is subsituted by the event.
   The two above .csv are also saved in the folder of the experiments.
   A splits_metadata contains the split_version and all its parameters to avoid recalculating a split with same parameters.
   If new detections are run, all files of the split_files folder are erased : new splits will have to be run.

5) Image preprocessing
    Choose a transform function of image_preprocessing.py or design a new one. It will be performed before model finetuning and cosine similarity.

6) Data augmentation
    It is implemented only when the model is fine-tuned. Choose
8) Model Finetuning
   If FINETUNE=True (in pipeline.py), a feature extractor is fine-tuned and evaluated.

   If you choose the architecture in pipeline.py default last_layer and frozen layer will be used. If you put None, fill pipeline.yaml feature_extractor dict with the name of the layers. You can look into layer_candidates.ipynb. You can choose any module of a feature extractor, noted as module.submodule.subsubmodule. The output of a module is the output of its last submodule. The last frozen layer is the name of the last untrained module. The output of the module referred to as last layer is taken as features before the classification head.

   You can choose data augmentation, 
9) Cosine similarity


Improvements : add data augmentation to cosine similarity


