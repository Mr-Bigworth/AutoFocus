# [AutoFocus: Uncertainty-Aware Active Visual Search for GUI Grounding](https://arxiv.org/abs/)

[//]: # (<a href="https://arxiv.org/abs/2505.00684"><img src="https://img.shields.io/badge/arXiv-2505.00684-b31b1b.svg" height=20.5></a>)

[//]: # ([Tiange Luo]&#40;https://tiangeluo.github.io/&#41;, [Lajanugen Logeswaran]&#40;https://lajanugen.github.io/&#41;&dagger;, [Justin Johnson]&#40;https://web.eecs.umich.edu/~justincj&#41;&dagger;, [Honglak Lee]&#40;https://web.eecs.umich.edu/~honglak/&#41;&dagger;)

We release our ScreenSpot-Pro code for both UI-TARS and Qwen2.5-VL. All hyperparameters and prompts are not carefully tuned. 
## ScreenSpot-Pro

Please first download the data from ScreenSpot-Pro [Hugging Face](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/tree/main) and put `images` and `annotations` folders under the same directory with code. Then, launch inference servers for different models (commands are listed below; the model names and ports have already been mapped inside the code). Finally, run `bash run_ss_pro_xxx.sh`.

You can use `summarize_results.py` to output ScreenSpot-Pro results categorically, following the order presented in our Table 1.
One Example:
```bash
python summarize_results.py results/qwen25vl_AutoFocus.json 

# output: 
# results/qwen25vl_AutoFocus_72b_PPL.json 85.1 & 32.9 & 60.2 & 77.7 & 40.1 & 62.3 & 74.1 & 33.9 & 64.1 & 91.0 & 39.4 & 69.4 & 87.6 & 60.8 & 81.6 & 78.8 & 31.4 & 57.4 & 82.1 & 38.1 & 65.6 1545
```

You can turn on `--debug` inside `eval_screenspot_pro_RegionFocus.py` to save intermediate RegionFocus step images, such as image-as-map stars for judgment, zoom-ins, and projecting zoomed-in predictions back onto the original input.

<details>
<summary>Command for launching Qwen2.5-VL-72B </summary>

Please first install https://github.com/QwenLM/Qwen-Agent. 

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
vllm serve Qwen/Qwen2.5-VL-72B-Instruct --port 8300  --dtype bfloat16   --limit-mm-per-prompt '{"images": 5}'   --tensor-parallel-size 8
```

</details>


## Citation Information

If you find our code or paper useful, please consider citing:

```
@article{autofocus2026,
      title={AutoFocus: Uncertainty-Aware Active Visual Search for GUI Grounding},
      author={Anonymous},
      year={2026},
}
```

### Acknowledge
This codebase is partially based on [RegionFocus](https://github.com/tiangeluo/RegionFocus). Many thanks!
