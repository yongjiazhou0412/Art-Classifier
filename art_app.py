import streamlit as st
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import asyncio
import nest_asyncio
import os

# # 修复事件循环问题
# nest_asyncio.apply()

# 创建新的事件循环
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 设置页面标题和图标
st.set_page_config(
    page_title="Art Style Classifier",
    page_icon="🎨",
    layout="wide"
)

# 加载模型
def load_model(model_path, num_classes):
    # 加载 EfficientNet-B0 模型
    # model = models.efficientnet_b0(pretrained=False)
    model = models.efficientnet_b0(weights=None)
    # 替换最后一层，使输出维度与艺术风格类别数相匹配
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 设置为评估模式
    return model

# 读取 artists.csv 文件
artists_df = pd.read_csv('artists.csv')

# 提取 name 列，并转换为列表
class_names = artists_df['name'].tolist()

# 图片预处理
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5041, 0.4451, 0.3724], std=[0.2741, 0.2647, 0.2603])
    ])
    image = transform(image).unsqueeze(0)  # 增加 batch 维度
    return image

# 加载模型
model_path = "best_model_artworks_efficientnet.pth"
num_classes = len(class_names)
model = load_model(model_path, num_classes)

# 页面标题
st.title("🎨 Art Style Classifier")
st.markdown("""
    <style>
    .big-font {
        font-size: 20px !important;
    }
    </style>
    <div class="big-font">
        Upload a picture, and the model will predict its art style.
    </div>
    """, unsafe_allow_html=True)

# 文件上传组件
uploaded_file = st.file_uploader("Upload a picture...", type=["jpg", "jpeg", "png"], help="Support JPG、JPEG、PNG formate")

if uploaded_file is not None:
    # 打开图片
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Picture')
    # # 预处理图片
    # input_tensor = preprocess_image(image)
    
    # 预处理图片
    with st.spinner("Model is predicting..."):
        input_tensor = preprocess_image(image)  
        # 使用模型进行预测
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=1)
            predicted_class = torch.argmax(probabilities).item()
            
    # # 显示预测结果
    # st.write(f"Predicted Style: **{class_names[predicted_class]}**")
    # st.write(f"probabilities: {probabilities[predicted_class].item() * 100:.2f}%")
    
    # 显示预测结果
    st.subheader("Prediction Result")
    predicted_class = torch.argmax(probabilities).item()
    st.success(f"Predict Style: **{class_names[predicted_class]}**")
    st.write(f"Confidence: {probabilities[predicted_class].item() * 100:.2f}%")
    
    # 显示 Top-3 预测结果
    st.subheader("Top-3 Prediction Result")
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    for i, (prob, idx) in enumerate(zip(top3_prob, top3_indices)):
        st.write(f"{i + 1}. {class_names[idx]} - {prob.item() * 100:.2f}%")

    # Draw heatmap for top 10 classes
    st.subheader("Top 10 Class Probability Heatmap")
    top10_prob, top10_indices = torch.topk(probabilities, 10)  # Get top 10 classes
    top10_class_names = [class_names[i] for i in top10_indices]  # Get class names
    top10_probabilities = top10_prob.numpy()  # Convert to numpy array

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(top10_probabilities.reshape(1, -1), cmap="YlGnBu", aspect="auto")
    plt.colorbar(label="Probability")
    plt.xticks(np.arange(len(top10_class_names)), top10_class_names, rotation=45, ha="right")
    plt.title("Top 10 Class Probability Distribution")
    plt.xlabel("Art Style")
    plt.ylabel("Probability")
    st.pyplot(plt)

# 侧边栏
with st.sidebar:
    # st.header("About")
    # st.markdown("""
    #     这是一个基于 EfficientNet-B0 的艺术风格分类器。
    #     - 训练数据: [Artists Dataset]
    #     - 模型: EfficientNet-B0
    #     - 开发者: [Your Name]
    # """)
    # st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
        1. Upload a picture.
        2. Check the prediction result.
        3. View the prediction result and heatmap.
    """)
    
        # Add a dropdown for model selection
    model_option = st.selectbox(
        "Choose a model to predict:",
        ("VGG", "ViT", "EfficientNet")
    )
