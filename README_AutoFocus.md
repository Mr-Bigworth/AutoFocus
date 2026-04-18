# AutoFocus: Uncertainty-Aware Active Visual Search for GUI Grounding 

<a href="#"><img src="https://img.shields.io/badge/ECCV_2026-Submission_xx-b31b1b.svg" height=20.5></a>

[//]: # ([cite_start]**Anonymous Authors** [cite: 9])

[//]: # ([cite_start]*ECCV 2026 Submission #xxxx* [cite: 10])

Welcome to the official repository for **AutoFocus**. [cite_start]This project introduces an uncertainty-aware active visual search methodology for GUI grounding[cite: 6, 7]. [cite_start]Our approach is designed to address key challenges in UI agent navigation, primarily linguistic underspecification, semantic ambiguity [cite: 21][cite_start], and perception limits that lead to a "quantization gap" in coordinate discretization[cite: 43, 46]. [cite_start]By utilizing a refinement-aggregation loop, AutoFocus mitigates localized failures for elements with minimal pixel footprints[cite: 47].

[cite_start]We provide the grounding, judgment, and aggregation prompts used across several state-of-the-art vision-language models, including Qwen2.5-VL [cite: 50][cite_start], GTA1-Qwen [cite: 78][cite_start], UI-Tars [cite: 84][cite_start], and UI-Venus[cite: 87].

## AutoFocus Pipeline

Our method relies on three main prompting stages to ensure precise visual grounding:
1. [cite_start]**Coordinate Prediction**: Initial grounding using model-specific context prompts[cite: 49].
2. [cite_start]**Active Refinement (Judgment)**: Utilizing a red star as a visual marker to verify if the localized position is correct and precise for the given instruction[cite: 94, 95, 97].
3. [cite_start]**Aggregation**: Analyzing 5 potential points and selecting the single best match[cite: 100, 102, 103].

---

### [cite_start]Prompts for AutoFocus [cite: 48]

Below are the exact prompt templates utilized in our pipeline. You can inject your specific `{instruction}`, `{width}`, and `{height}` variables where applicable.

<details>
<summary>1. Grounding Prompts for Coordinate Prediction</summary>

[cite_start]**For Qwen2.5-VL / UI-Tars** [cite: 50, 84]
> [cite_start]Output the most relevant point in the image corresponding to the instruction {instruction} with grounding. [cite: 73, 85]

[cite_start]**For GTA1-Qwen** [cite: 78]
> You are an expert UI element locator. [cite_start]Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. [cite: 79] The image resolution is height {height} and width {width}. [cite_start]For elements with area, return the center point. [cite: 80] [cite_start]Output the coordinate pair exactly: (x,y) [cite: 81]
> [cite_start]{instruction} [cite: 82]

[cite_start]**For UI-Venus** [cite: 87]
> Outline the position corresponding to the instruction: {instruction}. [cite_start]The output should be only [x1,y1,x2,y2] [cite: 88]
</details>

<details>
<summary>2. Judgement Prompt for Active Refinement</summary>

[cite_start]*Note: This prompt requires overlaying a red star on the image as a visual marker before passing it to the model.* [cite: 95]

> [cite_start]Given the instruction: {instruction}, I highlighted a red star on the image. [cite: 96] [cite_start]Is this red star position correct and precise for the instruction? [cite: 97] [cite_start]Answer "YES" if it accurately identifies the icon or element mentioned in the instruction. [cite: 98] [cite_start]Answer "NO" if its incorrect or imprecise. [cite: 99]
</details>

<details>
<summary>3. Aggregation Prompt</summary>

*Note: This prompt is used after identifying multiple potential coordinate candidates.*

> [cite_start]In these images, I have identified 5 potential points that might match the instruction: {instruction}. [cite: 102] [cite_start]Carefully analyze each point and select the ONE that best matches the instruction. [cite: 103] [cite_start]Provide your final answer in this format: [cite: 104] [cite_start]"Selected point: #" where # is the number of the best point. [cite: 105]
</details>

---

## [cite_start]Limitations Discussed [cite: 20]

* [cite_start]**Multi-modal Density Peaks**: Instructional ambiguity can cause multiple visually similar controls to satisfy a query (e.g., several "Edit" buttons) [cite: 22][cite_start], leading to multi-modal density peaks in the generated probability field[cite: 23].
* [cite_start]**Semantic Reasoning**: Agents may localize semantically valid regions that deviate from specific ground-truth annotations [cite: 24][cite_start], highlighting the need for future GUI-Agents to possess stronger semantic understanding, reasoning capabilities, and interactive solutions[cite: 25].
* [cite_start]**Hallucinatory Grounding**: UI components with low contrast frequently exceed the model's effective receptive field[cite: 45]. [cite_start]This triggers high-perplexity outputs that can lead to hallucinatory grounding[cite: 45].

## Citation Information

If you find our methodology, prompts, or analysis useful, please consider citing our ECCV submission:

```bibtex
@article{autofocus2026,
      title={AutoFocus: Uncertainty-Aware Active Visual Search for GUI Grounding},
      author={Anonymous},
      year={2026},
}