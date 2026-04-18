#    !/bin/bash

CUDA_VISIBLE_DEVICES=5 python eval_screenspot_pro_AutoFocus.py  \
        --model_type qwen25vl_AutoFocus_72b_PPL  \
        --model_name_or_path /root/models/Qwen2.5-VL-72B-Instruct \
        --screenspot_imgs "/root/Lens_VisionReasoner/RegionFocus-GUI-grounding/RegionFocus-main/images"  \
        --screenspot_test "/root/Lens_VisionReasoner/RegionFocus-GUI-grounding/RegionFocus-main/annotations"  \
        --task "all" \
        --language "en" \
        --gt_type "positive" \
        --log_path "./results/qwen25vl_AutoFocus_72b_PPL.json" \
        --checkpoint_path "./results_mid/qwen25vl_AutoFocus_72b_PPL.json" \
        --inst_style "instruction"
