export CUDA_VISIBLE_DEVICES=6 
python ./train/t2v_train.py \
--text_encoder_name ../hub/AI-ModelScope/t5-v1_1-xxl \
--dataset video_only \
--video_folder ./dataset/t2v_video \
--num_frames 17 \
--max_image_size 256 \
--train_batch_size 1 \
--model_max_length 300 \
--dataloader_num_workers 1 \
--sample_rate 1 \
--data_path ./dataset/video \
--save_video_path ./result/video/ \
--ae stabilityai/sd-vae-ft-mse
