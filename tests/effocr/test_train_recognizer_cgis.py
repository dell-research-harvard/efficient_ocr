from efficient_ocr import EffOCR

def test_recog_train():

    data_json = "../../data/coco_ex.json"
    config_json = "../../config/config_ex.json"
    data_dir = '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ocr_datasets/locca/images'

    effocr = EffOCR(
        data_json, data_dir, config_json, recog_only=True
    )

    effocr.train(
        target='char_recognition',
        batch_size=128,
        lr=2e-3,
        num_epochs=20,
        char_trans_version=4,
        num_passes=10,
        ascender=True,
        latin_suggested_augs=True,
        # ready_to_go_data_dir_path_word='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/fullsym_locc_dict_words_paired_silver/images',
        ready_to_go_data_dir_path_char='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ocr_char_crops/locca',
        output_dir='./effocr_package_test_locca',
        # render_dict_word='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress/silver_dpd/full_wordlist_effocr.txt',
        render_dict_char='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/jake_github_repos/ocr-as-retrieval/english_charsets/allchars.txt',
        font_dir_path='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/jake_github_repos/ocr-as-retrieval/english_font_files',
        wandb_project='effocr_package_test_locca',
    )

if __name__ == '__main__':
    test_recog_train()