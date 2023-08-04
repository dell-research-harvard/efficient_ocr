
CUDA_VISIBLE_DEVICES=2 python /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/github_repos/efficient_ocr/src/efficient_ocr/recognition/recognizer_training/train.py \
                                    --font_dir_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/extra_font_styles/ \
									--root_dir_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/test_dir_word_char/images \
									--word_dict /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/jake_github_repos/ocr-as-retrieval/edgenextSmall_effocrGold_onlySynth_augV3/ref.txt \
									--ascender \
									--train_ann_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/small_anno_file.json  \
									--val_ann_path  /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/small_anno_file.json \
									--test_ann_path /mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/small_anno_file.json \
                                    --train_mode "word" \
									--run_name testing_word_char \
									--auto_model_timm mobilenetv3_small_050  \
									--m 4 \
									--batch_size 64 \
									--num_epochs 1 \
									--num_passes 1 \
									--lr 1e-3 \
									--test_at_end \
									--imsize 224 \
									--infer_hardneg_k 8 \
									--weight_decay  0.0005  \
									--k 8 \
									--dec_lr_factor 1 \
									--lr_schedule \
									--epoch_viz_dir "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/epoch_viz_dir/" \
									--checkpoint "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/ocr-as-retrieval/mobilenet_silver_25_expanded_dict_final_es_filteredEasy/enc_best.pth" \
									--int_eval_steps 2000 \
                                    --default_font_name "Noto" \
                                    # --hns_txt_path "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/ocr-as-retrieval/mobilenet_silver_25_expanded_dict_final_es_filteredEasy/hns.txt" \
