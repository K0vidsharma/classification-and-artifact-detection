from transformers import AutoModel, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
import json
import os
from PIL import Image

model_id = "HuggingFaceM4/idefics2-8b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map = "auto")

adapter_config_path = "IDEFICS_DocVQA/checkpoint-25/adapter_config.json"
folder_dir = "IDEFICS_DocVQA/checkpoint-25/"
offload = "./offload_folder"
with open(adapter_config_path, 'r') as f:
    adapter_config_dict = json.load(f)

model.load_adapter(folder_dir, device_map='auto', offload_folder=offload)
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
# print(prompt)

model.eval()
image=Image.open("task-2-image/bird 4_8.png")


inputs = processor(text=[prompt.strip()], images=[image], return_tensors="pt", padding=True).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=10000)
print(generated_ids)
gen_text = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
print(gen_text)
