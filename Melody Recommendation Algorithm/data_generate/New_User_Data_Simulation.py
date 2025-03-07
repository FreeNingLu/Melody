import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import pickle
import soundfile as sf
import logging
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置日志记录
logging.basicConfig(
    filename='user_data_processing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # 调试阶段设置为 DEBUG，生产环境可调整为 INFO
)

logging.info("开始生成用户交互数据。")

# 定义文件路径
song_embeddings_file = r"E:/Melody/recommend/song_embeddings.pkl"
style_embeddings_file = r"E:/Melody/recommend/style_embeddings.pkl"
user_listen_data_file = r"E:/Melody/recommend/user_listen_data1.csv"
user_features_file = r"E:/Melody/recommend/user_data.csv"  # 用户信息文件
output_directory = r"E:/Melody/recommend"

# 定义缓存文件路径
song_lengths_cache_file = os.path.join(output_directory, 'song_lengths.pkl')
audio_mapping_cache_file = os.path.join(output_directory, 'song_audio_mapping.pkl')

# 定义音频文件所在目录（用于读取歌曲长度）
music_directory = r"E:/Melody/Music"  #歌曲目录

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"使用的设备：{device}")

# 加载歌曲数据+++++
logging.info(f"加载歌曲嵌入数据 from '{song_embeddings_file}'。")
song_df = pd.read_pickle(song_embeddings_file)

# 确认必要的列存在
required_columns = ['song_id', 'song_name', 'artist', 'audio_embedding']
missing_columns = [col for col in required_columns if col not in song_df.columns]
if missing_columns:
    raise ValueError(f"歌曲数据中缺少以下列：{missing_columns}，请确保 'song_embeddings.pkl' 包含这些列。")
logging.info(f"成功加载 {len(song_df)} 首歌曲的数据，包含必要的列。")

# 使用现有的 song_id
song_ids = song_df['song_id'].tolist()
song_names = song_df['song_name'].tolist()  # 保留 song_names 以供后续使用
# song_length_dict = {}  # 将在后续步骤中填充

logging.info("使用现有的 'song_id'。")

# 加载或生成音频文件映射
if os.path.exists(audio_mapping_cache_file):
    logging.info(f"加载缓存的音频文件映射 from '{audio_mapping_cache_file}'。")
    with open(audio_mapping_cache_file, 'rb') as f:
        song_audio_mapping = pickle.load(f)
else:
    logging.info(f"开始遍历 '{music_directory}' 以创建音频文件映射。")
    song_audio_mapping = {}
    for root, dirs, files in os.walk(music_directory):
        for file in files:
            if file.lower().endswith('.wav'):
                # 假设文件名为 'song_name.wav' 或其他格式
                song_name = os.path.splitext(file)[0]
                song_audio_mapping[song_name.strip().upper()] = os.path.join(root, file)
    logging.info("音频文件映射创建完成。")
    # 保存缓存
    with open(audio_mapping_cache_file, 'wb') as f:
        pickle.dump(song_audio_mapping, f)
    logging.info(f"缓存音频文件映射保存到 '{audio_mapping_cache_file}'。")

# 定义函数获取歌曲长度（使用映射）
def get_song_length(song_id, song_name):
    song_name_key = song_name.strip().upper()
    if song_name_key in song_audio_mapping:
        file_path = song_audio_mapping[song_name_key]
        try:
            data, samplerate = sf.read(file_path)
            duration = len(data) / samplerate  # 以秒为单位
            logging.info(f"获取歌曲 '{song_name}' 的实际长度: {duration:.2f} 秒。")
            return duration
        except Exception as e:
            logging.error(f"无法读取音频文件 '{file_path}' 以获取长度。错误信息: {e}")
    # 如果无法找到或读取音频文件，随机生成长度
    random_length = random.randint(180, 300)
    logging.warning(f"无法获取歌曲 '{song_name}' 的实际长度，随机生成长度: {random_length} 秒。")
    return random_length

# 尝试加载歌曲长度缓存
if os.path.exists(song_lengths_cache_file):
    with open(song_lengths_cache_file, 'rb') as f:
        song_length_dict = pickle.load(f)
    logging.info(f"已加载缓存的歌曲长度 from '{song_lengths_cache_file}'。")
else:
    song_length_dict = {}
    logging.info("开始生成歌曲长度。")
    for idx, row in tqdm(song_df.iterrows(), total=song_df.shape[0], desc="Calculating song lengths"):
        song_id = row['song_id']
        song_name = row['song_name']
        length = get_song_length(song_id, song_name)
        song_length_dict[song_id] = length
    logging.info("歌曲长度生成完成。")
    # 保存歌曲长度缓存
    with open(song_lengths_cache_file, 'wb') as f:
        pickle.dump(song_length_dict, f)
    logging.info(f"歌曲长度已缓存到 '{song_lengths_cache_file}'。")

# 添加 'song_length' 到 song_df
song_df['song_length'] = song_df['song_id'].map(song_length_dict)
logging.info("将 'song_length' 添加到 song_df 中。")

# 提取必要的字段
song_lengths = song_df['song_length'].tolist()
audio_embeddings = song_df['audio_embedding'].tolist()

# 创建 song_embedding_dict 和 song_length_dict
song_embedding_dict = dict(zip(song_df['song_id'], song_df['audio_embedding']))
# song_length_dict 已经存在
logging.info("创建歌曲嵌入和长度的字典映射。")

# 加载 style 嵌入
logging.info(f"加载 style 嵌入数据 from '{style_embeddings_file}'。")
with open(style_embeddings_file, 'rb') as f:
    style_embeddings = pickle.load(f)
logging.info("成功加载 style 嵌入数据。")

# 用户配置
logging.info("从 user_features.csv 加载用户信息。")
if os.path.exists(user_features_file):
    user_features_df = pd.read_csv(user_features_file)
    logging.info(f"成功加载用户特征数据，共有 {len(user_features_df)} 个用户。")
    # 打印列名以确认
    print("Columns in user_features_df:", user_features_df.columns.tolist())
    # 打印前几行数据以进一步确认
    print("First few rows in user_features_df:")
    print(user_features_df.head())
else:
    logging.error(f"用户特征文件 '{user_features_file}' 不存在。")
    raise FileNotFoundError(f"用户特征文件 '{user_features_file}' 不存在。")

# 获取新用户ID列表
new_user_ids = user_features_df['user_id'].tolist()

# 检查是否存在重复的 user_id
if os.path.exists(user_listen_data_file):
    existing_listen_data = pd.read_csv(user_listen_data_file)
    existing_user_ids = existing_listen_data['user_id'].unique().tolist()
    duplicate_user_ids = set(new_user_ids) & set(existing_user_ids)
    if duplicate_user_ids:
        logging.error(f"用户ID {duplicate_user_ids} 已存在于现有数据中。")
        raise ValueError(f"用户ID {duplicate_user_ids} 已存在于现有数据中。")
    logging.info("未发现重复的用户ID，可以继续生成数据。")
else:
    logging.info("未找到现有的用户数据，所有用户ID均为新用户。")

# 提取用户特征
style_column_name = 'style'  # 根据实际列名修改
if style_column_name not in user_features_df.columns:
    logging.error(f"用户特征文件中缺少 '{style_column_name}' 列。")
    raise KeyError(f"用户特征文件中缺少 '{style_column_name}' 列。")
user_style = dict(zip(new_user_ids, user_features_df[style_column_name].astype(str).str.strip().str.upper()))
genders = dict(zip(new_user_ids, user_features_df['gender']))
locations = dict(zip(new_user_ids, user_features_df['location']))
logging.info("提取用户的 style 类型、性别和位置。")

# 计算所有 style 嵌入的平均值，作为默认嵌入
default_embedding = np.mean(list(style_embeddings.values()), axis=0)
default_embedding = default_embedding / np.linalg.norm(default_embedding)

# 生成用户偏好向量
logging.info("生成用户偏好向量。")
user_preferences = {}
for idx, user_id in enumerate(new_user_ids):
    style_type = user_style[user_id].strip().upper()  # 去除空格并转换为大写
    # 检查 style 类型是否在 style_embeddings 中
    if style_type in style_embeddings:
        base_pref = style_embeddings[style_type]
        logging.info(f"用户 {user_id} 的 style 类型 '{style_type}' 找到对应的嵌入。")
    else:
        logging.warning(f"用户 {user_id} 的 style 类型 '{style_type}' 不在 style 嵌入中，使用默认嵌入。")
        base_pref = default_embedding
    # 添加随机噪声
    random_noise = np.random.normal(0, 0.05, size=base_pref.shape)
    user_pref = base_pref + random_noise
    user_pref = user_pref / np.linalg.norm(user_pref)
    user_preferences[user_id] = user_pref
    # 打印前几个用户的偏好向量用于验证
    if idx < 5:
        logging.debug(f"用户 {user_id} 的偏好向量: {user_pref}")
        print(f"用户 {user_id} 的偏好向量: {user_pref}")
logging.info("用户偏好向量生成完成。")

# 将歌曲嵌入转换为 PyTorch 张量，并移动到设备上，指定数据类型为 float32
for song_id in song_embedding_dict:
    song_embedding_dict[song_id] = torch.tensor(
        song_embedding_dict[song_id],
        device=device,
        dtype=torch.float32
    )

# 初始化交互记录列表
interaction_records = []

# 生成用户交互数据
logging.info("开始生成用户交互数据。")
start_date = datetime.now() - timedelta(days=30)
base_share_prob = 0.2
base_comment_prob = 0.1

similarity_list = []
like_list = []

for user_id in tqdm(new_user_ids, desc="Generating interactions", unit="user"):
    user_pref = user_preferences[user_id]
    # 将用户偏好转换为 PyTorch 张量，并移动到设备上，指定数据类型为 float32
    user_pref_tensor = torch.tensor(user_pref, device=device, dtype=torch.float32)
    active_hour = random.choice(range(24))
    for _ in range(100):
        song_id = random.choice(song_ids)
        song_row = song_df[song_df['song_id'] == song_id].iloc[0]
        song_name = song_row['song_name']
        artist = song_row['artist']
        song_length = song_length_dict[song_id]
        song_embedding_tensor = song_embedding_dict[song_id]

        # 确保 song_embedding_tensor 为 float32 类型
        song_embedding_tensor = song_embedding_tensor.float()

        # 归一化和计算相似度
        song_embedding_norm = song_embedding_tensor / torch.norm(song_embedding_tensor)
        user_pref_tensor_norm = user_pref_tensor / torch.norm(user_pref_tensor)

        # 相似度计算
        similarity = torch.dot(user_pref_tensor_norm, song_embedding_norm).item() + np.random.normal(0, 0.4)
        like_probability = 1 / (1 + np.exp(-6 * (similarity - 0.4)))

        # 调试日志
        logging.debug(f"用户 {user_id} 与歌曲 {song_id} 的相似度: {similarity:.4f}, like_probability: {like_probability:.4f}")
        similarity_list.append(similarity)
        like_list.append(int(random.random() < like_probability))

        like = int(random.random() < like_probability)

        # 模拟听歌时长和时间戳
        if random.random() < 0.1:
            listen_duration = random.uniform(0, song_length * 0.2)
        else:
            if like:
                listen_duration = random.uniform(song_length * 0.5, song_length)
            else:
                listen_duration = random.uniform(30, song_length * 0.5)
        # 随机生成一个日期时间
        random_seconds = random.randint(0, 30 * 24 * 60 * 60)
        random_date = start_date + timedelta(seconds=random_seconds)
        try:
            timestamp = random_date.replace(hour=active_hour, minute=random.randint(0,59), second=random.randint(0,59))
        except ValueError as e:
            # 防止随机生成的小时、分钟、秒超出范围
            logging.error(f"生成时间戳时出错: {e}")
            timestamp = random_date

        # 模拟交互行为
        share_prob = base_share_prob
        comment_prob = base_comment_prob
        download, share, comment = [int(like and random.random() < prob) for prob in [0.4, share_prob, comment_prob]]

        # 更新用户偏好
        if like:
            user_pref_tensor = user_pref_tensor + 0.1 * (song_embedding_norm - user_pref_tensor)
            user_pref_tensor = user_pref_tensor / torch.norm(user_pref_tensor)

        # 将用户偏好张量转换回 NumPy 数组，便于后续使用
        user_pref = user_pref_tensor.cpu().numpy()

        # 记录交互数据，添加 'artist' 和 'song_length'
        interaction_records.append([
            user_id, song_name, song_id, artist,
            listen_duration, timestamp, like, download, share, comment
        ])

    # 更新用户偏好字典
    user_preferences[user_id] = user_pref

logging.info("用户交互数据生成完成。")

# 创建交互数据 DataFrame
interaction_df = pd.DataFrame(
    interaction_records,
    columns=[
        'user_id', 'song_name', 'song_id', 'artist',
        'listen_duration', 'timestamp', 'like', 'download', 'share', 'comment'
    ]
)