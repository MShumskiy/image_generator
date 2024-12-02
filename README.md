# Image Generator

This project consists of a free highly customizable image generator.

## Usage

### Structure
This project is structured to to run in declarative paradigm by making use of the files in the **configs** folder.
**configs folder**:
- **config.json** - serves as the run configs source.
- **prompt.txt** - serves as the run prompt source.
**configs.json example**
    "model_id": "black-forest-labs/FLUX.1-schnell", - huggingface model directory
    "lora": true, - whether to enable lora or not
    "enable_tiling": true, - whether to enable tiling
    "enable_slicing": true, - whether to enable slicing
    "enable_sequential_cpu_offload": true, - whether to enable sequential cpu offload
    "safety_checker": false, - whether to enable censorhip
    "enable_attention_slicing": true, - whether to enable attention slicing
    "num_inference_steps":1, - num of inference steps per image
    "guidance_scale":8, - the guidance scale for the image generation
    "gen_type":"target", - type of seed iteration, if interval, an iteration will be performed for the provided range, if not interval will iterate through the list of seeds provided
    "seeds":[1] - list of seeds to iterate according to the preceding parameter. **must always be a list**
### Output 
The output will consist in the following structure:
- Output folder:
  - images_i - all images here generated have the same hyperparameters apart from num of inference steps and guidance scale
    - (guidance_scale)_(num_inference_steps) - folder with the hyperparameters in the name fixed and its content will be the images iterated through the provided seeds list
      -image_seed_i - image corresponding to the seed i.
    - generation_configs.json - json of configs to which all the images where generated apart from guidance scale and num of inference steps
### Usage
The usage consists in altering the **configs.json** and **prompt.txt** and then running the **ImageGenerator** class as per the **test** notebook
  
