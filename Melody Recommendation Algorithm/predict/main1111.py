import requests
import pandas as pd
import os
import pickle
import soundfile as sf
import logging
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

############################################
#          1. 从后端接口获取用户信息并保存CSV
############################################

# 定义后端接口的URL和JWT Token，用于获取用户信息
url = "https://melody-go.com/melody/login/api/auth/profile"
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# 将token放入cookies中
cookies = {
    'token': token
}
headers = {
    "Content-Type": "application/json"
}
# 用户数据保存路径
csv_file = 'autodl-tmp/Data/user_data.csv'

try:
    # 向后端发送请求，获取用户信息
    response = requests.get(url, headers=headers, cookies=cookies)
    if response.status_code == 200:
        # 成功获取数据
        data = response.json()
        user_hash = data.get('id', '')
        style = data.get('style', '')
        gender = data.get('gender', '')
        location = data.get('country', '')

        # 如果CSV不存在则创建一个空的DataFrame
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=['user_id', 'user_hash', 'style', 'gender', 'location'])

        # 检查用户是否已存在
        if user_hash in df['user_hash'].values:
            # 用户已存在则获取其 user_id
            user_id = df.loc[df['user_hash'] == user_hash, 'user_id'].iloc[0]
            print(f"用户已存在，user_id: {user_id}")
        else:
            # 如果不存在则为其分配新的 user_id
            if df.empty:
                user_id = 1
            else:
                user_id = df['user_id'].max() + 1

            new_row = pd.DataFrame({
                'user_id': [user_id],
                'user_hash': [user_hash],
                'style': [style],
                'gender': [gender],
                'location': [location]
            })

            df = pd.concat([df, new_row], ignore_index=True)
            print(f"分配新的 user_id: {user_id}")

        # 保存更新的数据到CSV
        df.to_csv(csv_file, index=False)
        print('数据已保存到 CSV 文件：', csv_file)
        print('当前用户信息：')
        print(df[df['user_id'] == user_id])
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print(f"响应内容：{response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求过程中出现异常：{e}")


############################################
#    2. 生成用户交互数据（模拟用户历史行为）
############################################

# 配置日志记录，用于调试和查看运行状态
logging.basicConfig(
    filename='user_data_processing.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info("开始生成用户交互数据。")

# 定义文件路径
song_embeddings_file = r"/root/autodl-tmp/recommend/song_embeddings.pkl"
style_embeddings_file = r"/root/autodl-tmp/recommend/style_embeddings.pkl"
user_listen_data_file = r"/root/autodl-tmp/recommend/user_listen_data.csv"
user_features_file = r"/root/autodl-tmp/recommend/user_data.csv"
output_directory = r"/root/autodl-tmp/recommend"

song_lengths_cache_file = os.path.join(output_directory, 'song_lengths.pkl')
audio_mapping_cache_file = os.path.join(output_directory, 'song_audio_mapping.pkl')

# 确定运行设备（GPU或CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"使用的设备：{device}")

# 加载歌曲嵌入数据
logging.info(f"加载歌曲嵌入数据 from '{song_embeddings_file}'。")
song_df = pd.read_pickle(song_embeddings_file)

# 检查歌曲数据中必要的列
required_columns = ['song_id', 'song_name', 'artist', 'audio_embedding']
missing_columns = [col for col in required_columns if col not in song_df.columns]
if missing_columns:
    raise ValueError(f"歌曲数据中缺少以下列：{missing_columns}")

logging.info(f"成功加载 {len(song_df)} 首歌曲的数据，包含必要的列。")


############################################
#  在此添加降维与聚类步骤
############################################

# 从song_df中提取所有的audio_embedding形成数组
audio_embeddings = np.stack(song_df['audio_embedding'].values)

# 使用PCA对音频嵌入降维，例如降到50维
pca_dim = 50
logging.info(f"对音频嵌入进行 PCA 降维到 {pca_dim} 维。")
pca = PCA(n_components=pca_dim)
reduced_embeddings = pca.fit_transform(audio_embeddings)

# 对降维后的结果进行KMeans聚类，比如分成10个簇
num_clusters = 10
logging.info(f"对降维后的嵌入进行 KMeans 聚类到 {num_clusters} 个簇。")
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_embeddings)

# 将聚类结果加入到song_df
song_df['cluster_id'] = cluster_labels
logging.info("已为每首歌曲分配 cluster_id。")


############################################
# 加载音频映射文件，计算歌曲长度
############################################

# 加载音频文件映射（song_name -> audio_file_path）
if os.path.exists(audio_mapping_cache_file):
    logging.info(f"加载缓存的音频文件映射 from '{audio_mapping_cache_file}'。")
    with open(audio_mapping_cache_file, 'rb') as f:
        song_audio_mapping = pickle.load(f)
else:
    logging.error(f"音频文件映射文件 '{audio_mapping_cache_file}' 不存在。")
    raise FileNotFoundError(f"音频文件映射文件 '{audio_mapping_cache_file}' 不存在。")

def get_song_length(song_id, song_name):
    """
    尝试从音频文件中获取歌曲长度，如果失败则随机生成一个长度。
    """
    song_name_key = song_name.strip().upper()
    if song_name_key in song_audio_mapping:
        file_path = song_audio_mapping[song_name_key]
        try:
            data, samplerate = sf.read(file_path)
            duration = len(data) / samplerate
            logging.info(f"获取歌曲 '{song_name}' 的实际长度: {duration:.2f} 秒。")
            return duration
        except Exception as e:
            logging.error(f"无法读取音频文件 '{file_path}' 以获取长度。错误信息: {e}")
    random_length = random.randint(180, 300)
    logging.warning(f"无法获取歌曲 '{song_name}' 的实际长度，随机生成长度: {random_length} 秒。")
    return random_length

# 加载或生成歌曲长度缓存
if os.path.exists(song_lengths_cache_file):
    with open(song_lengths_cache_file, 'rb') as f:
        song_length_dict = pickle.load(f)
    logging.info(f"已加载缓存的歌曲长度 from '{song_lengths_cache_file}'。")
else:
    song_length_dict = {}
    logging.info("开始生成歌曲长度。")
    for idx, row in tqdm(song_df.iterrows(), total=song_df.shape[0], desc="Calculating song lengths"):
        s_id = row['song_id']
        s_name = row['song_name']
        length = get_song_length(s_id, s_name)
        song_length_dict[s_id] = length
    logging.info("歌曲长度生成完成。")

    # 保存歌曲长度到缓存文件
    with open(song_lengths_cache_file, 'wb') as f:
        pickle.dump(song_length_dict, f)
    logging.info(f"歌曲长度已缓存到 '{song_lengths_cache_file}'。")

# 将歌曲长度加入 song_df
song_df['song_length'] = song_df['song_id'].map(song_length_dict)
logging.info("将 'song_length' 添加到 song_df 中。")

# 提取必要的数据
song_ids = song_df['song_id'].tolist()
song_embedding_dict = dict(zip(song_df['song_id'], song_df['audio_embedding']))

# 加载风格（style）嵌入数据
logging.info(f"加载 style 嵌入数据 from '{style_embeddings_file}'。")
with open(style_embeddings_file, 'rb') as f:
    style_embeddings = pickle.load(f)
logging.info("成功加载 style 嵌入数据。")

# 加载用户特征
logging.info("从 user_features.csv 加载用户信息。")
if os.path.exists(user_features_file):
    user_features_df = pd.read_csv(user_features_file)
    logging.info(f"成功加载用户特征数据，共有 {len(user_features_df)} 个用户。")
    print("Columns in user_features_df:", user_features_df.columns.tolist())
    print("First few rows in user_features_df:")
    print(user_features_df.head())
else:
    logging.error(f"用户特征文件 '{user_features_file}' 不存在。")
    raise FileNotFoundError(f"用户特征文件 '{user_features_file}' 不存在。")

new_user_ids = user_features_df['user_id'].tolist()

# 检查是否有重复的用户ID（避免重复生成数据）
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

# 用户特征中的style列名
style_column_name = 'style'
if style_column_name not in user_features_df.columns:
    logging.error(f"用户特征文件中缺少 '{style_column_name}' 列。")
    raise KeyError(f"用户特征文件中缺少 '{style_column_name}' 列。")

# 构建用户特征字典
user_style = dict(zip(new_user_ids, user_features_df[style_column_name].astype(str).str.strip().str.upper()))
genders = dict(zip(new_user_ids, user_features_df['gender']))
locations = dict(zip(new_user_ids, user_features_df['location']))

# 默认嵌入（所有style平均）
default_embedding = np.mean(list(style_embeddings.values()), axis=0)
default_embedding = default_embedding / np.linalg.norm(default_embedding)

# 为用户生成偏好向量（根据style嵌入 + 随机扰动）
logging.info("生成用户偏好向量。")
user_preferences = {}
for idx, u_id in enumerate(new_user_ids):
    style_type = user_style[u_id].strip().upper()
    if style_type in style_embeddings:
        base_pref = style_embeddings[style_type]
        logging.info(f"用户 {u_id} 的 style 类型 '{style_type}' 找到对应的嵌入。")
    else:
        logging.warning(f"用户 {u_id} 的 style 类型 '{style_type}' 不在 style 嵌入中，使用默认嵌入。")
        base_pref = default_embedding
    random_noise = np.random.normal(0, 0.05, size=base_pref.shape)
    user_pref = base_pref + random_noise
    user_pref = user_pref / np.linalg.norm(user_pref)
    user_preferences[u_id] = user_pref
    if idx < 5:
        logging.debug(f"用户 {u_id} 的偏好向量: {user_pref}")
        print(f"用户 {u_id} 的偏好向量: {user_pref}")
logging.info("用户偏好向量生成完成。")

# 将歌曲嵌入转为Tensor
for s_id in song_embedding_dict:
    song_embedding_dict[s_id] = torch.tensor(
        song_embedding_dict[s_id],
        device=device,
        dtype=torch.float32
    )

# 模拟用户交互记录
interaction_records = []
logging.info("开始生成用户交互数据。")
start_date = datetime.now() - timedelta(days=30)
base_share_prob = 0.2
base_comment_prob = 0.1

for user_id in tqdm(new_user_ids, desc="Generating interactions", unit="user"):
    user_pref = user_preferences[user_id]
    user_pref_tensor = torch.tensor(user_pref, device=device, dtype=torch.float32)
    active_hour = random.choice(range(24))
    for _ in range(100):
        song_id = random.choice(song_ids)
        song_row = song_df[song_df['song_id'] == song_id].iloc[0]
        song_name = song_row['song_name']
        artist = song_row['artist']
        song_length = song_length_dict[song_id]
        song_embedding_tensor = song_embedding_dict[song_id].float()

        # 计算相似度（用户偏好与歌曲embedding）
        song_embedding_norm = song_embedding_tensor / torch.norm(song_embedding_tensor)
        user_pref_tensor_norm = user_pref_tensor / torch.norm(user_pref_tensor)
        similarity = torch.dot(user_pref_tensor_norm, song_embedding_norm).item() + np.random.normal(0, 0.4)
        like_probability = 1 / (1 + np.exp(-6 * (similarity - 0.4)))
        like = int(random.random() < like_probability)

        # 根据是否喜欢决定听歌时长分布
        if random.random() < 0.1:
            listen_duration = random.uniform(0, song_length * 0.2)
        else:
            if like:
                listen_duration = random.uniform(song_length * 0.5, song_length)
            else:
                listen_duration = random.uniform(30, song_length * 0.5)

        # 生成随机交互时间（过去30天内）
        random_seconds = random.randint(0, 30 * 24 * 60 * 60)
        random_date = start_date + timedelta(seconds=random_seconds)
        try:
            timestamp = random_date.replace(hour=active_hour, minute=random.randint(0,59), second=random.randint(0,59))
        except ValueError as e:
            logging.error(f"生成时间戳时出错: {e}")
            timestamp = random_date

        # 模拟下载、分享、评论行为
        share_prob = base_share_prob
        comment_prob = base_comment_prob
        download, share, comment = [int(like and random.random() < prob) for prob in [0.4, share_prob, comment_prob]]

        # 如果用户喜欢这首歌，则稍微更新用户偏好
        if like:
            user_pref_tensor = user_pref_tensor + 0.1 * (song_embedding_norm - user_pref_tensor)
            user_pref_tensor = user_pref_tensor / torch.norm(user_pref_tensor)

        user_pref = user_pref_tensor.cpu().numpy()

        # 记录一条交互数据
        interaction_records.append([
            user_id, song_name, song_id, artist,
            listen_duration, timestamp, like, download, share, comment
        ])

    # 更新用户偏好字典
    user_preferences[user_id] = user_pref

logging.info("用户交互数据生成完成。")

# 将交互数据保存到CSV
interaction_df = pd.DataFrame(
    interaction_records,
    columns=[
        'user_id', 'song_name', 'song_id', 'artist',
        'listen_duration', 'timestamp', 'like', 'download', 'share', 'comment'
    ]
)

if os.path.exists(user_listen_data_file):
    interaction_df.to_csv(user_listen_data_file, mode='a', header=False, index=False)
    logging.info(f"已将 {len(interaction_df)} 条交互数据追加到 '{user_listen_data_file}'。")
else:
    interaction_df.to_csv(user_listen_data_file, mode='w', header=True, index=False)
    logging.info(f"已创建并保存 {len(interaction_df)} 条交互数据到 '{user_listen_data_file}'。")

print(f"已为 {len(new_user_ids)} 个新用户生成交互数据，追加到 {user_listen_data_file}")
logging.info("用户交互数据处理完成。")


############################################
#           3. 使用Transformer模型进行推荐
############################################

print(f'Using device: {device}')

# 定义模型相关参数（需与训练时一致）
USER_EMBEDDING_DIM = 64
SONG_EMBEDDING_DIM = 256
AUDIO_EMBEDDING_DIM = 1024
TEXT_EMBEDDING_DIM = 1024
BEHAVIOR_FEATURE_DIM = 5
HIDDEN_DIM = 128
MAX_SEQ_LEN = 100
NUM_HEADS = 12
NUM_LAYERS = 6
FEEDFORWARD_DIM = 3072

# 加载预处理文件和模型文件
with open('user_encoder_filtered.pkl', 'rb') as f:
    user_encoder_filtered = pickle.load(f)
with open('song_encoder.pkl', 'rb') as f:
    song_encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('song_id_to_name.pkl', 'rb') as f:
    song_id_to_name = pickle.load(f)

num_users = len(user_encoder_filtered.classes_)
num_songs = len(song_encoder.classes_)

# 定义模型结构，与训练时保持一致
class SequenceModel(nn.Module):
    def __init__(self, num_users, num_songs, user_embedding_dim, song_embedding_dim,
                 audio_embedding_dim, text_embedding_dim, behavior_feature_dim, hidden_dim,
                 max_seq_len=100, num_heads=12, num_layers=6, feedforward_dim=3072):
        super(SequenceModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, song_embedding_dim)
        self.input_dim = song_embedding_dim + behavior_feature_dim + audio_embedding_dim + text_embedding_dim
        self.d_model = 768  # Transformer维度
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = nn.Embedding(max_seq_len, self.d_model)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads, dim_feedforward=feedforward_dim, dropout=0.1, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.d_model + user_embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, song_ids, behavior_features, audio_embeddings, text_embeddings, sequence_lengths):
        user_emb = self.user_embedding(user_ids)
        song_emb = self.song_embedding(song_ids)
        transformer_input = torch.cat((song_emb, behavior_features, audio_embeddings, text_embeddings), dim=2)
        transformer_input = self.input_projection(transformer_input)

        batch_size, seq_len, _ = transformer_input.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=transformer_input.device).unsqueeze(0).expand(batch_size, seq_len)
        position_emb = self.positional_encoding(position_ids)
        transformer_input = transformer_input + position_emb

        key_padding_mask = torch.arange(seq_len, device=transformer_input.device).unsqueeze(0).expand(batch_size, seq_len) >= sequence_lengths.unsqueeze(1)
        transformer_output = self.transformer_encoder(transformer_input, src_key_padding_mask=key_padding_mask)
        transformer_output = self.dropout(transformer_output)

        last_outputs = transformer_output[torch.arange(batch_size), sequence_lengths - 1]
        x = torch.cat((user_emb, last_outputs), dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.output(x)
        x = self.sigmoid(x)

        return x

# 实例化模型，并加载训练好的参数
model = SequenceModel(
    num_users=num_users,
    num_songs=num_songs,
    user_embedding_dim=USER_EMBEDDING_DIM,
    song_embedding_dim=SONG_EMBEDDING_DIM,
    audio_embedding_dim=AUDIO_EMBEDDING_DIM,
    text_embedding_dim=TEXT_EMBEDDING_DIM,
    behavior_feature_dim=BEHAVIOR_FEATURE_DIM,
    hidden_dim=HIDDEN_DIM,
    max_seq_len=MAX_SEQ_LEN,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    feedforward_dim=FEEDFORWARD_DIM
).to(device)

model.load_state_dict(torch.load('trained_sequence_model.pth', map_location=device))
model.eval()

# 加载用户交互数据和歌曲嵌入数据，用于预测
user_data_df = pd.read_csv('/root/autodl-tmp/Data/user_listen_data.csv')
song_embeddings_df = pd.read_pickle('/root/autodl-tmp/Data/song_embeddings.pkl')

# 清洗数据（移除无效值）
song_embeddings_df = song_embeddings_df.dropna(subset=['text_embedding', 'audio_embedding'])
data_df = pd.merge(user_data_df, song_embeddings_df, on='song_id', how='inner')
data_df = data_df.dropna()

# 编码用户与歌曲ID
data_df['user_id_encoded'] = user_encoder_filtered.transform(data_df['user_id'])
data_df['song_id_encoded'] = song_encoder.transform(data_df['song_id'])

# 对听歌时长进行归一化
data_df['listen_duration_norm'] = scaler.transform(data_df[['listen_duration']])

# 提取时间特征
data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
data_df['hour'] = data_df['timestamp'].dt.hour
data_df['day_of_week'] = data_df['timestamp'].dt.dayofweek

# 按用户和时间排序
data_df = data_df.sort_values(['user_id_encoded', 'timestamp'])

# 假设对用户ID=1进行预测（可根据需要修改）
user_ids_to_predict = [1]
results_all_users = []

for user_id in user_ids_to_predict:
    if user_id in user_encoder_filtered.classes_:
        user_id_encoded = user_encoder_filtered.transform([user_id])[0]
    else:
        print(f"User ID {user_id} not found in the dataset.")
        continue

    # 获取用户的历史数据
    user_data = data_df[data_df['user_id_encoded'] == user_id_encoded].copy()
    user_data = user_data.sort_values('timestamp')

    if user_data.empty:
        print(f"No data found for user ID {user_id}")
        continue

    # 截取用户历史，不超过 MAX_SEQ_LEN - 1 条
    if len(user_data) > MAX_SEQ_LEN - 1:
        user_history = user_data.iloc[-(MAX_SEQ_LEN - 1):]
    else:
        user_history = user_data

    # 候选歌曲 = 用户未听过的所有歌曲
    all_song_ids = song_encoder.classes_
    user_song_ids = user_data['song_id'].unique()
    candidate_song_ids = np.setdiff1d(all_song_ids, user_song_ids)

    candidate_songs_df = song_embeddings_df[song_embeddings_df['song_id'].isin(candidate_song_ids)].copy()
    candidate_songs_df['song_id_encoded'] = song_encoder.transform(candidate_songs_df['song_id'])

    # 使用用户历史平均值填充候选歌曲的行为特征
    avg_listen_duration_norm = user_history['listen_duration_norm'].mean()
    avg_hour = user_history['hour'].mean()
    avg_day_of_week = user_history['day_of_week'].mean()

    candidate_songs_df['listen_duration_norm'] = avg_listen_duration_norm
    candidate_songs_df['hour'] = avg_hour
    candidate_songs_df['day_of_week'] = avg_day_of_week

    # 准备用户历史特征
    song_ids_history = torch.tensor(user_history['song_id_encoded'].values, dtype=torch.long)
    listen_duration_history = torch.tensor(user_history['listen_duration_norm'].values, dtype=torch.float32)
    hour_history = torch.tensor(user_history['hour'].values, dtype=torch.float32)
    day_of_week_history = torch.tensor(user_history['day_of_week'].values, dtype=torch.float32)
    behavior_features_history = torch.stack([listen_duration_history, hour_history, day_of_week_history], dim=1)
    audio_embeddings_history = torch.tensor(np.stack(user_history['audio_embedding'].values), dtype=torch.float32)
    text_embeddings_history = torch.tensor(np.stack(user_history['text_embedding'].values), dtype=torch.float32)

    # 准备候选歌曲特征
    song_ids_candidate = torch.tensor(candidate_songs_df['song_id_encoded'].values, dtype=torch.long)
    listen_duration_candidate = torch.tensor(candidate_songs_df['listen_duration_norm'].values, dtype=torch.float32)
    hour_candidate = torch.tensor(candidate_songs_df['hour'].values, dtype=torch.float32)
    day_of_week_candidate = torch.tensor(candidate_songs_df['day_of_week'].values, dtype=torch.float32)
    behavior_features_candidate = torch.stack([listen_duration_candidate, hour_candidate, day_of_week_candidate], dim=1)
    audio_embeddings_candidate = torch.tensor(np.stack(candidate_songs_df['audio_embedding'].values), dtype=torch.float32)
    text_embeddings_candidate = torch.tensor(np.stack(candidate_songs_df['text_embedding'].values), dtype=torch.float32)

    # 创建批次数据，将用户历史与候选歌曲拼接起来形成输入序列
    batch_size = len(candidate_songs_df)
    sequence_length = len(user_history) + 1

    song_ids_batch = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    behavior_features_batch = torch.zeros((batch_size, sequence_length, BEHAVIOR_FEATURE_DIM), dtype=torch.float32)
    audio_embeddings_batch = torch.zeros((batch_size, sequence_length, AUDIO_EMBEDDING_DIM), dtype=torch.float32)
    text_embeddings_batch = torch.zeros((batch_size, sequence_length, TEXT_EMBEDDING_DIM), dtype=torch.float32)
    sequence_lengths_batch = torch.full((batch_size,), sequence_length, dtype=torch.long)

    # 将用户历史扩展到batch，并填入batch张量中
    song_ids_history_expanded = song_ids_history.unsqueeze(0).expand(batch_size, -1)
    behavior_features_history_expanded = behavior_features_history.unsqueeze(0).expand(batch_size, -1, -1)
    audio_embeddings_history_expanded = audio_embeddings_history.unsqueeze(0).expand(batch_size, -1, -1)
    text_embeddings_history_expanded = text_embeddings_history.unsqueeze(0).expand(batch_size, -1, -1)

    song_ids_batch[:, :len(user_history)] = song_ids_history_expanded
    behavior_features_batch[:, :len(user_history), :] = behavior_features_history_expanded
    audio_embeddings_batch[:, :len(user_history), :] = audio_embeddings_history_expanded
    text_embeddings_batch[:, :len(user_history), :] = text_embeddings_history_expanded

    # 在序列最后一个位置放置候选歌曲的数据
    song_ids_batch[:, -1] = song_ids_candidate
    behavior_features_batch[:, -1, :] = behavior_features_candidate
    audio_embeddings_batch[:, -1, :] = audio_embeddings_candidate
    text_embeddings_batch[:, -1, :] = text_embeddings_candidate

    user_ids_batch = torch.full((batch_size,), user_id_encoded, dtype=torch.long)

    # 将数据移到设备（GPU或CPU）
    user_ids_batch = user_ids_batch.to(device)
    song_ids_batch = song_ids_batch.to(device)
    behavior_features_batch = behavior_features_batch.to(device)
    audio_embeddings_batch = audio_embeddings_batch.to(device)
    text_embeddings_batch = text_embeddings_batch.to(device)
    sequence_lengths_batch = sequence_lengths_batch.to(device)

    # 使用模型进行预测
    with torch.no_grad():
        outputs = model(user_ids_batch, song_ids_batch, behavior_features_batch, audio_embeddings_batch, text_embeddings_batch, sequence_lengths_batch)

    predicted_scores = outputs.squeeze().cpu().numpy()
    candidate_song_ids = candidate_songs_df['song_id'].values

    results = pd.DataFrame({
        'song_id': candidate_song_ids,
        'score': predicted_scores
    })

    # 获取Top40的推荐歌曲
    top_40 = results.sort_values(by='score', ascending=False).head(40)
    top_40['song_name'] = top_40['song_id'].map(song_id_to_name)

    results_all_users.append({
        'user_id': user_id,
        'recommendations': top_40[['song_id', 'song_name', 'score']].values.tolist()
    })

# 打印推荐结果
for result in results_all_users:
    print(f"用户 ID {result['user_id']} 最可能喜欢的歌曲：")
    for song in result['recommendations']:
        print(f"歌曲 ID: {song[0]}, 歌曲名称: {song[1]}, 评分: {song[2]}")
