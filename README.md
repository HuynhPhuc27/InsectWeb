# InsectWeb
Multimodal insect classification system using Cross-Attention mechanism to integrate image features (ViT) and text descriptions (BERT). Integrated with a Django web interface for real-time identification..
Insect identification plays a critical role in various fields, particularly in precision agriculture, pest monitoring, and biodiversity conservation. In agriculture, accurate identification enables early detection of pests, supporting effective control measures and minimizing crop damage. In biological conservation, identifying insect species contributes to monitoring rare species and assessing ecosystem health
However, traditional image-based systems often struggle with lighting conditions, varying angles, or complex backgrounds. This project proposes a multimodal approach that combines visual data with textual descriptions (such as color, size, and behavior) to provide additional semantic context, enhancing overall accuracy and generalization.
Model Architecture: * Vision: Vision Transformer (ViT).
  + Text: BERT (Bidirectional Encoder Representations from Transformers).
  + Fusion: One-way Cross-Attention mechanism.
The research primarily utilizes the IP102 dataset, a large-scale benchmark for insect recognition: Contains over 75,000 images across 102 species of common pests.
The project is developed and deployed using the following technologies:
  + Deep Learning Framework: PyTorch (utilizing pretrained ViT and BERT backbones).
  + Web Framework: Django for a user-friendly interface.
  + Database: MySQL to manage insect data and system logs.
