import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Idefics2ForConditionalGeneration, Idefics2Config
from transformers.models.idefics2.configuration_idefics2 import Idefics2VisionConfig

import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
import os, json
from transformers import TrainingArguments, Trainer
from transformers import AutoProcessor
from peft import LoraConfig
from transformers import BitsAndBytesConfig
processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules='.*(text_model|modality_projection|perceiver_resampler).(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
    use_dora=True,
    init_lora_weights="gaussian"
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b",
                                                         device_map='auto',
                                                         torch_dtype=torch.float16,
                                                         quantization_config=bnb_config,
)

model.add_adapter(lora_config)
model.enable_adapters()

id2label= {
    1: "Dental anomalies in mammals",
    2: "Anatomically incorrect paw structures",
    3: "Improper fur direction flows",
    4: "Unrealistic eye reflections",
    5: "Misshapen ears or appendages",
    6: "Impossible foreshortening in animal bodies",
    7: "Anatomically impossible joint configurations",
    8: "Misaligned body panels",
    9: "Biological asymmetry errors",
    10: "Misaligned bilateral elements in animal faces",
    11: "Discontinuous surfaces",
    12: "Non-manifold geometries in rigid structures",
    13: "Floating or disconnected components",
    14: "Asymmetric features in naturally symmetric objects",
    15: "Irregular proportions in mechanical components",
    16: "Abruptly cut off objects",
    17: "Incorrect wheel geometry",
    18: "Implausible aerodynamic structures",
    19: "Impossible mechanical joints",
    20: "Inconsistent material properties",
    21: "Metallic surface artifacts",
    22: "Impossible mechanical connections",
    23: "Inconsistent scale of mechanical parts",
    24: "Physically impossible structural elements",
    25: "Scale inconsistencies within single objects",
    26: "Jagged edges in curved structures",
    27: "Inconsistent object boundaries",
    28: "Blurred boundaries in fine details",
    29: "Scale inconsistencies within the same object class",
    30: "Artificial noise patterns in uniform surfaces",
    31: "Unrealistic specular highlights",
    32: "Missing ambient occlusion",
    33: "Spatial relationship errors",
    34: "Over-sharpening artifacts",
    35: "Aliasing along high-contrast edges",
    36: "Random noise patterns in detailed areas",
    37: "Loss of fine detail in complex structures",
    38: "Artificial enhancement artifacts",
    39: "Unnatural pose artifacts",
    40: "Systematic color distribution anomalies",
    41: "Color coherence breaks",
    42: "Unnatural color transitions",
    43: "Resolution inconsistencies within regions",
    44: "Glow or light bleed around object boundaries",
    45: "Unnaturally glossy surfaces",
    46: "Unnatural lighting gradients",
    47: "Texture bleeding between adjacent regions",
    48: "Texture repetition patterns",
    49: "Over-smoothing of natural textures",
    50: "Regular grid-like artifacts in textures",
    51: "Repeated element patterns",
    52: "Frequency domain signatures",
    53: "Ghosting effects",
    54: "Inconsistent shadow directions",
    55: "Multiple light source conflicts",
    56: "Multiple inconsistent shadow sources",
    57: "Dramatic lighting that defies natural physics",
    58: "Distorted window reflections",
    59: "Fake depth of field",
    60: "Artificial depth of field in object presentation",
    61: "Incorrect reflection mapping",
    62: "Incorrect perspective rendering",
    63: "Depth perception anomalies",
    64: "Cinematization effects",
    65: "Excessive sharpness in certain image regions",
    66: "Artificial smoothness",
    67: "Movie-poster-like composition of ordinary scenes",
    68: "Exaggerated characteristic features",
    69: "Incorrect skin tones",
    70: "Synthetic material appearance"
}





system_message="""You are my assistant to analyze the image. Analyze the image and tell if it has these anomalies: 
1. Dental anomalies in mammals: Teeth structures that deviate from normal anatomical patterns.  
2. Anatomically incorrect paw structures: Unrealistic shapes or alignments of animal paws.  
3. Improper fur direction flows: Hair patterns that defy natural growth.  
4. Unrealistic eye reflections: Impractical or overly dramatic highlights in eyes.  
5. Misshapen ears or appendages: Oddly shaped or proportioned body parts.  
6. Impossible foreshortening in animal bodies: Unnatural shortening of animal limbs or torsos.  
7. Anatomically impossible joint configurations: Joint positions that cannot occur naturally.  
8. Misaligned body panels: Parts of the body that appear out of place.  
9. Biological asymmetry errors: Uneven features where symmetry is expected.  
10. Misaligned bilateral elements in animal faces: Disproportionate placement of eyes, ears, or nostrils.  
11. Discontinuous surfaces: Breaks or interruptions in surface continuity.  
12. Non-manifold geometries in rigid structures: Impossible or non-contiguous geometries in objects.  
13. Floating or disconnected components: Elements that appear detached or unsupported.  
14. Asymmetric features in naturally symmetric objects: Lack of symmetry in objects meant to be balanced.  
15. Irregular proportions in mechanical components: Mechanical parts with unrealistic size ratios.  
16. Abruptly cut off objects: Structures that end unnaturally or without tapering.  
17. Incorrect wheel geometry: Wheels with distorted or impractical shapes.  
18. Implausible aerodynamic structures: Designs that wouldn't function in real airflow.  
19. Impossible mechanical joints: Connections that defy engineering principles.  
20. Inconsistent material properties: Materials behaving or appearing unnaturally.  
21. Metallic surface artifacts: Unrealistic reflections or surface defects in metals.  
22. Impossible mechanical connections: Structural joins that wouldn’t hold under real conditions.  
23. Inconsistent scale of mechanical parts: Size mismatches within a single object.  
24. Physically impossible structural elements: Features that defy physics or practicality.  
25. Scale inconsistencies within single objects: Variability in the scale of features within one item.  
26. Jagged edges in curved structures: Rough or pixelated contours on smooth shapes.  
27. Inconsistent object boundaries: Edges that vary in sharpness or definition.  
28. Blurred boundaries in fine details: Lack of clarity in intricate areas.  
29. Scale inconsistencies within the same object class: Size differences where uniformity is expected.  
30. Artificial noise patterns in uniform surfaces: Non-natural graininess in flat textures.  
31. Unrealistic specular highlights: Overly bright or exaggerated surface shines.  
32. Missing ambient occlusion: Lack of natural shading in recessed areas.  
33. Spatial relationship errors: Misaligned or conflicting object placements.  
34. Over-sharpening artifacts: Excessive edge definition causing unnatural looks.  
35. Aliasing along high-contrast edges: Jagged lines where smooth transitions are expected.  
36. Random noise patterns in detailed areas: Grain or texture inconsistency in detailed regions.  
37. Loss of fine detail in complex structures: Simplified or blurred intricate designs.  
38. Artificial enhancement artifacts: Over-processed visual elements.  
39. Unnatural pose artifacts: Poses or stances that defy physics or biology.  
40. Systematic color distribution anomalies: Odd patterns in color gradients or transitions.  
41. Color coherence breaks: Sudden changes in color where consistency is expected.  
42. Unnatural color transitions: Gradients that appear forced or unrealistic.  
43. Resolution inconsistencies within regions: Uneven sharpness or detail levels.  
44. Glow or light bleed around object boundaries: Unnatural halos or luminance near edges.  
45. Unnaturally glossy surfaces: Excessively shiny finishes that look synthetic.  
46. Unnatural lighting gradients: Light spreads that don't mimic real-world behavior.  
47. Texture bleeding between adjacent regions: Overlapping or smeared textures.  
48. Texture repetition patterns: Noticeable tiling or duplication of textures.  
49. Over-smoothing of natural textures: Excessive blending that erases texture detail.  
50. Regular grid-like artifacts in textures: Visible grids disrupting natural surfaces.  
51. Repeated element patterns: Overly repetitive features or motifs.  
52. Frequency domain signatures: Visible processing artifacts in high-frequency details.  
53. Ghosting effects: Semi-transparent, duplicated parts of objects.  
54. Inconsistent shadow directions: Shadows that don’t align with a single light source.  
55. Multiple light source conflicts: Shadows suggesting conflicting light angles.  
56. Multiple inconsistent shadow sources: Shadows with varying intensity or origin.  
57. Dramatic lighting that defies natural physics: Unrealistically intense or directed light.  
58. Distorted window reflections: Impractical reflection patterns on glass surfaces.  
59. Fake depth of field: Unconvincing simulated focus effects.  
60. Artificial depth of field in object presentation: Depth simulation that lacks realism.  
61. Incorrect reflection mapping: Reflections that fail to align with object positioning.  
62. Incorrect perspective rendering: Misalignment of object angles or depths.  
63. Depth perception anomalies: Inconsistent depth cues in a scene.  
64. Cinematization effects: Overly dramatic or theatrical visual styling.  
65. Excessive sharpness in certain image regions: Overemphasis on specific details.  
66. Artificial smoothness: Overly polished surfaces that lack texture.  
67. Movie-poster-like composition of ordinary scenes: Over-stylized layouts for mundane settings.  
68. Exaggerated characteristic features: Overemphasis on particular traits.  
69. Incorrect skin tones: Colors that don't resemble realistic skin.  
70. Synthetic material appearance: Textures or finishes that seem fake or unnatural.  

Analyze which options which are found in picture if present and answer in the specified json format. I will give you two answer examples for reference
Example 1: {'Artifacts annotation': [
  {'artifacts_caption': "The man's hand is distorted.",
   'artifacts_class': '2'}],
 'Other artifacts caption': 'The man misses a leg in the generated image.'}

Example 2: {'Artifacts annotation': [{'artifacts_caption': 'deformed hand and racket, repeated hand',
   'artifacts_class': '2'}], 
 'Other artifacts caption': 'None'}
"""

conversation=[
    {
      "role":"system",
       "content":[
           {"type":"text","text":system_message}
       ]
    },
    {

      "role": "user",
      "content": [
          
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(prompt)




class ArtifactDataset(Dataset):
    def __init__(self, images_dir, json_dir,transform=None):
        super().__init__()
        self.transform = transform
        self.images_dir = images_dir
        self.json_dir = json_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.num_classes=70
    def __len__(self):
        return len(self.image_files)

    def json2token(self, obj):
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                keys = obj.keys()
                output=''
                for k in keys:
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k])
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item) for item in obj]
            )
        else:
            obj = str(obj)
            return obj

    def __getitem__(self, idx):
        # Get the image file name
        image_file = self.image_files[idx]
        
        # Load the image
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Load the corresponding JSON file
        gt_json_sequence=''
        json_file_path = os.path.join(self.json_dir, image_file.replace('.png', '.json').replace('.jpg', '.json').replace('.jpeg', '.json'))
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
            gt_json={'Artifacts annotation': [{'artifacts_caption':artifact['artifacts_caption'],'artifacts_class':artifact['artifacts_class']} for artifact in data['Artifacts annotation']], 'Other artifacts caption':data['Other artifacts caption']}
            gt_json_sequence=self.json2token(gt_json)
        return image, gt_json_sequence

# Example usage
# transform = transforms.Compose([
#     transforms.ToTensor(),transforms.Resize((32,32))])

images_dir = "./task-2-image/"
json_dir = "./final json"

train_dataset = ArtifactDataset(images_dir, json_dir,transform=None)

def train_collate_fn(examples):
    """
    Prepares a batch of training data for a model that processes images and text.

    Args:
        examples (list): A list of tuples where each tuple contains an image and the corresponding ground truth.

    Returns:
        tuple: A tuple containing:
            - input_ids (torch.Tensor): Tokenized input IDs of the text prompts.
            - attention_mask (torch.Tensor): Attention mask for the input IDs.
            - pixel_values (torch.Tensor): Preprocessed pixel values of the images.
            - image_sizes (list): List of original sizes of the images.
            - labels (torch.Tensor): Labels for training, with padding token IDs replaced by -100.
    """
    images = []
    texts = []
    for example in examples:
        image, ground_truth = example
        # if image.max() > 1.0:
        #     image = image / 255.0  # Normalize image to [0, 1]
        images.append(image)
        # TODO: in the future we can replace this by processor.apply_chat_template
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)+" "+ground_truth+processor.tokenizer.eos_token
       # prompt = f"[INST] <image>\n[\INST] {ground_truth}"
        texts.append(prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=2000, return_tensors="pt",do_rescale=False)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    labels = batch["labels"]

    return batch




training_args = TrainingArguments(
    output_dir = "IDEFICS_DocVQA",
    learning_rate = 2e-4,
    fp16 = True,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps = 8,
    dataloader_pin_memory = False,
    save_total_limit = 3,
    save_strategy = "steps",
    eval_steps = 10,
    save_steps = 50,
    max_steps = 50,
    logging_steps = 5,
    remove_unused_columns = False,
    push_to_hub=False,
    label_names = ["labels"],
    load_best_model_at_end = False,
    report_to = "none"
)
trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = train_collate_fn,
    train_dataset = train_dataset,

)

trainer.train()