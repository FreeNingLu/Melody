import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm.auto import tqdm  # 导入 tqdm 进度条库

# =========================
# 数据加载和预处理
# =========================

# 设置设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载用户行为数据和歌曲嵌入数据
user_data = pd.read_csv('/root/autodl-tmp/recommend/user_listen_data.csv')  # 用户听歌行为数据
song_embeddings = pd.read_pickle('/root/autodl-tmp/Data/song_embeddings.pkl')  # 歌曲的文本和音频嵌入

# 处理缺失值：删除缺少文本或音频嵌入的歌曲
song_embeddings = song_embeddings.dropna(subset=['text_embedding', 'audio_embedding'])

# 将用户数据与歌曲嵌入数据在 'song_id' 上进行内连接合并
data = pd.merge(user_data, song_embeddings, on='song_id', how='inner')
data = data.dropna()  # 删除任何仍有缺失值的行

# 用户编码：将用户ID从字符串编码为数值
user_encoder = LabelEncoder()
data['user_id_encoded'] = user_encoder.fit_transform(data['user_id'])
num_users = data['user_id_encoded'].nunique()  # 获取唯一用户数

# 歌曲编码：将歌曲ID从字符串编码为数值
song_encoder = LabelEncoder()
song_encoder.fit(song_embeddings['song_id'])
data['song_id_encoded'] = song_encoder.transform(data['song_id'])
num_songs = len(song_encoder.classes_)  # 获取唯一歌曲数

# 打印编码后的用户和歌曲信息
print("user_id_encoded min:", data['user_id_encoded'].min())
print("user_id_encoded max:", data['user_id_encoded'].max())
print("num_users:", num_users)

print("song_id_encoded min:", data['song_id_encoded'].min())
print("song_id_encoded max:", data['song_id_encoded'].max())
print("num_songs:", num_songs)

# 对听歌时长进行归一化处理
scaler = MinMaxScaler()
data['listen_duration_norm'] = scaler.fit_transform(data[['listen_duration']])

# 从时间戳中提取额外的时间特征
data['timestamp'] = pd.to_datetime(data['timestamp'])  # 转换为 datetime 类型
data['hour'] = data['timestamp'].dt.hour  # 提取小时特征
data['day_of_week'] = data['timestamp'].dt.dayofweek  # 提取星期几特征（0=星期一）

# 过滤掉仅有一次交互的用户，确保每个用户至少有多于一条的记录
user_counts = data['user_id_encoded'].value_counts()
users_with_multiple_interactions = user_counts[user_counts > 1].index
data_filtered = data[data['user_id_encoded'].isin(users_with_multiple_interactions)].copy()

# 更新用户编码以匹配过滤后的用户集
user_encoder_filtered = LabelEncoder()
data_filtered['user_id_encoded'] = user_encoder_filtered.fit_transform(data_filtered['user_id'])
num_users_filtered = data_filtered['user_id_encoded'].nunique()

print("Number of users after filtering:", num_users_filtered)

# 按用户ID和时间戳排序，以便构建时间序列数据
data_filtered = data_filtered.sort_values(['user_id_encoded', 'timestamp'])

# 计算数据中的最大序列长度（用于模型输入的序列长度限制）
max_seq_len_in_data = data_filtered.groupby('user_id_encoded').size().max()
print("Maximum sequence length in data:", max_seq_len_in_data)

# =========================
# 新增特征计算（播放次数与推广）
# =========================

# 计算每首歌曲的播放次数
data_filtered['play_count'] = data_filtered.groupby('song_id_encoded')['song_id_encoded'].transform('count')

# 计算每首歌曲的总播放时长
data_filtered['total_listen_duration'] = data_filtered.groupby('song_id_encoded')['listen_duration'].transform('sum')

# 增加推广播放数据
promotion_play_count = 100  # 预设的推广播放次数
average_listen_duration = data_filtered['listen_duration'].mean()  # 平均听歌时长

# 计算推广后的播放次数和总播放时长
data_filtered['play_count_promoted'] = data_filtered['play_count'] + promotion_play_count
data_filtered['total_listen_duration_promoted'] = data_filtered['total_listen_duration'] + (promotion_play_count * average_listen_duration)

# 归一化新增的收听数据特征
scaler_extra = MinMaxScaler()
data_filtered[['listen_duration', 'play_count_promoted', 'total_listen_duration_promoted']] = scaler_extra.fit_transform(
    data_filtered[['listen_duration', 'play_count_promoted', 'total_listen_duration_promoted']]
)

# 将归一化后的特征赋值回新列
data_filtered['listen_duration_norm'] = data_filtered['listen_duration']
data_filtered['play_count_promoted_norm'] = data_filtered['play_count_promoted']
data_filtered['total_listen_duration_promoted_norm'] = data_filtered['total_listen_duration_promoted']

# 检查新增特征是否存在
print("Columns after feature engineering:", data_filtered.columns)

# =========================
# 更新 song_embeddings 文件
# 将新增特征合并到 song_embeddings 中
# =========================

# 给 song_embeddings 添加 song_id_encoded 列
song_embeddings['song_id_encoded'] = song_encoder.transform(song_embeddings['song_id'])

# 计算每首歌曲的推广特征（取第一条记录的值，因为它们在整个数据集中应该是一致的）
song_listen_data = data_filtered.groupby('song_id_encoded').agg({
    'play_count_promoted_norm': 'first',
    'total_listen_duration_promoted_norm': 'first'
}).reset_index()

# 合并推广特征到 song_embeddings 中
song_embeddings = pd.merge(song_embeddings, song_listen_data, on='song_id_encoded', how='left')

# 填充缺失值（没有被过滤掉的歌曲的推广特征设为0）
song_embeddings['play_count_promoted_norm'] = song_embeddings['play_count_promoted_norm'].fillna(0)
song_embeddings['total_listen_duration_promoted_norm'] = song_embeddings['total_listen_duration_promoted_norm'].fillna(0)

# 检查合并后的 song_embeddings 是否包含新增的列
print("song_embeddings columns after merging promoted features:", song_embeddings.columns)

# 选择保存更新后的 song_embeddings.pkl 文件
song_embeddings.to_pickle('updated_song_embeddings.pkl')

# 将更新后的数据重新合并回 data 中，以便后续在数据集使用
# 为了避免重复的列名，合并前移除 data_filtered 中已经存在的 'play_count_promoted_norm' 和 'total_listen_duration_promoted_norm'
data_filtered_dropped = data_filtered.drop(columns=[
    'text_embedding', 
    'audio_embedding', 
    'play_count_promoted_norm', 
    'total_listen_duration_promoted_norm'
])

# 进行合并
data = pd.merge(
    data_filtered_dropped,
    song_embeddings[['song_id', 'song_id_encoded', 'text_embedding', 'audio_embedding', 
                    'play_count_promoted_norm', 'total_listen_duration_promoted_norm']], 
    on=['song_id_encoded', 'song_id'],
    how='inner'
)

# 检查合并后的 data 是否包含 'play_count_promoted_norm'
print("Columns in merged data:", data.columns)

# 确保 'play_count_promoted_norm' 列存在
if 'play_count_promoted_norm' not in data.columns:
    raise KeyError("'play_count_promoted_norm' not found in the merged data.")

# 删除任何包含缺失值的行，确保数据完整性
data = data.dropna()

# =========================
# 划分训练集和测试集
# =========================

# 使用分层抽样按用户编码划分训练集和测试集
train_data, test_data = train_test_split(
    data, test_size=0.2, stratify=data['user_id_encoded'], random_state=42
)

# 在划分后再次确认 'play_count_promoted_norm' 是否存在
print("Columns in train_data:", train_data.columns)
print("Columns in test_data:", test_data.columns)

# 确保训练集和测试集都包含 'play_count_promoted_norm'
if 'play_count_promoted_norm' not in train_data.columns:
    raise KeyError("'play_count_promoted_norm' not found in train_data.")

if 'play_count_promoted_norm' not in test_data.columns:
    raise KeyError("'play_count_promoted_norm' not found in test_data.")

# =========================
# 定义数据集
# =========================

class MusicDataset(Data.Dataset):
    """
    自定义数据集类，用于加载用户的听歌序列数据。
    每个样本包括用户ID、歌曲ID序列、行为特征序列、音频和文本嵌入序列，以及标签。
    """
    def __init__(self, data, max_seq_len=None):
        self.user_ids = []
        self.sequences = []
        self.sequence_lengths = []
        self.max_seq_len = max_seq_len

        # 按用户编码分组
        user_groups = data.groupby('user_id_encoded')
        # 使用 tqdm 进度条遍历用户组
        for user_id, group in tqdm(user_groups, desc="Building Dataset", total=user_groups.ngroups):
            # 如果序列长度超过最大长度，截取最后 max_seq_len 条记录
            if self.max_seq_len and len(group) > self.max_seq_len:
                group = group.iloc[-self.max_seq_len:]

            # 获取歌曲ID序列
            song_ids = torch.tensor(group['song_id_encoded'].values, dtype=torch.long)
            # 获取行为特征序列
            listen_duration = torch.tensor(group['listen_duration_norm'].values, dtype=torch.float32)
            hour = torch.tensor(group['hour'].values, dtype=torch.float32)
            day_of_week = torch.tensor(group['day_of_week'].values, dtype=torch.float32)
            # 获取音频和文本嵌入序列
            audio_embeddings = torch.tensor(np.stack(group['audio_embedding'].values), dtype=torch.float32)
            text_embeddings = torch.tensor(np.stack(group['text_embedding'].values), dtype=torch.float32)
            # 获取标签序列（假设 'like' 列表示对每首歌的喜好）
            labels = torch.tensor(group['like'].values, dtype=torch.float32)

            # 检查新增的推广特征列是否存在
            if 'play_count_promoted_norm' not in group.columns or 'total_listen_duration_promoted_norm' not in group.columns:
                raise KeyError("One of the promoted features is missing in the group.")

            # 获取推广特征序列
            play_count_promoted = torch.tensor(group['play_count_promoted_norm'].values, dtype=torch.float32)
            total_listen_duration_promoted = torch.tensor(group['total_listen_duration_promoted_norm'].values, dtype=torch.float32)

            # 构建行为特征序列（共5个特征）
            behavior_features = torch.stack([
                listen_duration,                  # 听歌时长
                hour,                             # 听歌时间的小时
                day_of_week,                      # 听歌时间的星期几
                play_count_promoted,             # 推广后的播放次数
                total_listen_duration_promoted    # 推广后的总听歌时长
            ], dim=1)

            # 构建样本字典
            sequence = {
                'song_ids': song_ids,
                'behavior_features': behavior_features,
                'audio_embeddings': audio_embeddings,
                'text_embeddings': text_embeddings,
                'labels': labels
            }

            # 将样本添加到数据集中
            self.user_ids.append(torch.tensor(user_id, dtype=torch.long))
            self.sequences.append(sequence)
            self.sequence_lengths.append(len(song_ids))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],       # 用户ID
            self.sequences[idx],      # 听歌序列及相关特征
            self.sequence_lengths[idx] # 序列长度
        )

def collate_fn(batch):
    """
    自定义的collate_fn函数，用于在DataLoader中批处理数据。
    它会对不同长度的序列进行填充，并将它们组合成批次。
    """
    user_ids = []
    song_ids = []
    behavior_features = []
    audio_embeddings = []
    text_embeddings = []
    labels = []
    sequence_lengths = []

    # 遍历每个样本，将各部分数据分离
    for user_id, sequence, seq_length in batch:
        user_ids.append(user_id)
        song_ids.append(sequence['song_ids'])
        behavior_features.append(sequence['behavior_features'])
        audio_embeddings.append(sequence['audio_embeddings'])
        text_embeddings.append(sequence['text_embeddings'])
        labels.append(sequence['labels'])
        sequence_lengths.append(seq_length)

    # 对序列进行填充，使得每个批次中的序列长度一致
    padded_song_ids = pad_sequence(song_ids, batch_first=True, padding_value=0)
    padded_behavior_features = pad_sequence(behavior_features, batch_first=True, padding_value=0)
    padded_audio_embeddings = pad_sequence(audio_embeddings, batch_first=True, padding_value=0)
    padded_text_embeddings = pad_sequence(text_embeddings, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    sequence_lengths = torch.tensor(sequence_lengths, dtype=torch.long)

    # 将用户ID堆叠成一个张量
    user_ids = torch.stack(user_ids)

    return (
        user_ids,
        padded_song_ids,
        padded_behavior_features,
        padded_audio_embeddings,
        padded_text_embeddings,
        padded_labels,
        sequence_lengths
    )

# =========================
# 定义模型
# =========================

# 定义行为特征的维度（共5个特征）
BEHAVIOR_FEATURE_DIM = 5

class SequenceModel(nn.Module):
    """
    基于Transformer的序列模型，用于预测用户对歌曲的喜好。
    输入包括用户嵌入、歌曲嵌入、行为特征、音频和文本嵌入。
    """
    def __init__(self, num_users, num_songs, user_embedding_dim, song_embedding_dim,
                 audio_embedding_dim, text_embedding_dim, behavior_feature_dim, hidden_dim,
                 max_seq_len=100, num_heads=12, num_layers=6, feedforward_dim=3072):
        super(SequenceModel, self).__init__()
        # 用户嵌入层，将用户ID映射为嵌入向量
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        # 歌曲嵌入层，将歌曲ID映射为嵌入向量
        self.song_embedding = nn.Embedding(num_songs, song_embedding_dim)
        # 定义输入特征的维度（歌曲嵌入 + 行为特征 + 音频嵌入 + 文本嵌入）
        self.input_dim = song_embedding_dim + behavior_feature_dim + audio_embedding_dim + text_embedding_dim
        # 定义Transformer的d_model维度
        self.d_model = 768  # 增大模型维度
        assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # 输入投影层，将输入特征映射到Transformer的d_model维度
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        # 位置编码层，用于添加序列中每个位置的位置信息
        self.positional_encoding = nn.Embedding(max_seq_len, self.d_model)
        # 定义Transformer编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=0.1)
        # 全连接层，用于将Transformer的输出和用户嵌入结合后进行进一步处理
        self.fc1 = nn.Linear(self.d_model + user_embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)  # 最终输出一个二分类概率
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_ids, song_ids, behavior_features, audio_embeddings, text_embeddings, sequence_lengths):
        """
        前向传播函数。
        """
        # 获取用户嵌入
        user_emb = self.user_embedding(user_ids)  # [batch_size, user_embedding_dim]

        # 获取歌曲嵌入
        song_emb = self.song_embedding(song_ids)  # [batch_size, seq_len, song_embedding_dim]

        # 合并所有特征
        transformer_input = torch.cat((song_emb, behavior_features, audio_embeddings, text_embeddings), dim=2)  # [batch_size, seq_len, input_dim]

        # 通过输入投影层
        transformer_input = self.input_projection(transformer_input)  # [batch_size, seq_len, d_model]

        # 添加位置编码
        batch_size, seq_len, _ = transformer_input.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=transformer_input.device).unsqueeze(0).expand(batch_size, seq_len)
        position_emb = self.positional_encoding(position_ids)  # [batch_size, seq_len, d_model]

        transformer_input = transformer_input + position_emb

        # 创建 key_padding_mask，形状为 [batch_size, seq_len]
        # True 表示需要被mask的部分（即序列的填充部分）
        key_padding_mask = torch.arange(seq_len, device=transformer_input.device).unsqueeze(0).expand(batch_size, seq_len) >= sequence_lengths.unsqueeze(1)

        # 通过Transformer编码器
        transformer_output = self.transformer_encoder(transformer_input, src_key_padding_mask=key_padding_mask)  # [batch_size, seq_len, d_model]
        transformer_output = self.dropout(transformer_output)

        # 取每个序列的最后一个有效时间步的输出
        last_outputs = transformer_output[torch.arange(batch_size), sequence_lengths - 1]  # [batch_size, d_model]

        # 拼接用户嵌入和Transformer的输出
        x = torch.cat((user_emb, last_outputs), dim=1)  # [batch_size, d_model + user_embedding_dim]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.output(x)  # [batch_size, 1]
        x = self.sigmoid(x)  # 将输出转换为概率

        return x

# =========================
# 超参数和训练准备
# =========================

# 定义嵌入和隐藏层的维度
USER_EMBEDDING_DIM = 64
SONG_EMBEDDING_DIM = 256
AUDIO_EMBEDDING_DIM = 1024
TEXT_EMBEDDING_DIM = 1024
HIDDEN_DIM = 128  

num_users = num_users_filtered  # 过滤后的用户数量
MAX_SEQ_LEN = max_seq_len_in_data  # 最大序列长度

# 使用自定义的 MusicDataset 类创建训练集和测试集
train_dataset = MusicDataset(train_data, max_seq_len=MAX_SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

test_dataset = MusicDataset(test_data, max_seq_len=MAX_SEQ_LEN)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 初始化模型并移动到指定设备
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
    num_heads=12,
    num_layers=6,
    feedforward_dim=3072
).to(device)

# =========================
# 定义训练函数
# =========================

def train_model(model, train_loader, device, epochs=20):
    """
    训练模型的函数。
    
    参数：
    - model: 待训练的模型
    - train_loader: 训练数据的DataLoader
    - device: 训练设备（CPU或GPU）
    - epochs: 训练的轮数
    """
    model.train()  # 设置模型为训练模式
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)  # AdamW优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # 学习率调度器

    for epoch in range(epochs):
        total_loss = 0
        # 使用 tqdm 进度条遍历训练集
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            user_ids, song_ids, behavior_features, audio_embeddings, text_embeddings, labels, sequence_lengths = batch

            # 将数据移动到指定设备
            user_ids = user_ids.to(device)
            song_ids = song_ids.to(device)
            behavior_features = behavior_features.to(device)
            audio_embeddings = audio_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)
            labels = labels.to(device)
            sequence_lengths = sequence_lengths.to(device)

            # 前向传播：获取模型输出
            outputs = model(user_ids, song_ids, behavior_features, audio_embeddings, text_embeddings, sequence_lengths)

            # 调整标签形状，取序列的最后一个时刻的label
            # 假设 'like' 列表示对最后一首歌的喜好
            labels = labels[:, -1].unsqueeze(1)  # [batch_size, 1]

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失
            total_loss += loss.item()

        # 更新学习率
        scheduler.step()
        # 计算并打印平均损失
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# =========================
# 定义评估函数
# =========================

def evaluate_model(model, test_loader, device):
    """
    评估模型在测试集上的性能。
    
    参数：
    - model: 已训练的模型
    - test_loader: 测试数据的DataLoader
    - device: 评估设备（CPU或GPU）
    """
    model.eval()  # 设置模型为评估模式
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    total_loss = 0

    with torch.no_grad():  # 关闭梯度计算
        # 使用 tqdm 进度条遍历测试集
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            user_ids, song_ids, behavior_features, audio_embeddings, text_embeddings, labels, sequence_lengths = batch

            # 将数据移动到指定设备
            user_ids = user_ids.to(device)
            song_ids = song_ids.to(device)
            behavior_features = behavior_features.to(device)
            audio_embeddings = audio_embeddings.to(device)
            text_embeddings = text_embeddings.to(device)
            labels = labels.to(device)
            sequence_lengths = sequence_lengths.to(device)

            # 前向传播：获取模型输出
            outputs = model(user_ids, song_ids, behavior_features, audio_embeddings, text_embeddings, sequence_lengths)

            # 调整标签形状，取序列的最后一个时刻的label
            labels = labels[:, -1].unsqueeze(1)  # [batch_size, 1]

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    # 计算并打印平均损失
    avg_loss = total_loss / len(test_loader)
    print(f'Evaluation Loss: {avg_loss:.4f}')

# =========================
# 开始训练和评估
# =========================

# 训练模型
train_model(model, train_loader, device, epochs=100)
# 评估模型
evaluate_model(model, test_loader, device)

# =========================
# 保存模型和相关组件
# =========================

# 保存训练好的模型参数
torch.save(model.state_dict(), 'trained_sequence_model.pth')
print("Model saved to 'trained_sequence_model.pth'.")

# 保存用户编码器，以便在推理阶段使用相同的编码
with open('user_encoder_filtered.pkl', 'wb') as f:
    pickle.dump(user_encoder_filtered, f)
print("User encoder saved to 'user_encoder_filtered.pkl'.")

# 保存歌曲编码器，以便在推理阶段使用相同的编码
with open('song_encoder.pkl', 'wb') as f:
    pickle.dump(song_encoder, f)
print("Song encoder saved to 'song_encoder.pkl'.")

# 保存归一化缩放器，以便在推理阶段对数据进行相同的归一化处理
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved to 'scaler.pkl'.")

# 保存歌曲ID到名称的映射（如果存在 'song_name' 列）
if 'song_name' in song_embeddings.columns:
    song_id_to_name = pd.Series(song_embeddings['song_name'].values, index=song_embeddings['song_id']).to_dict()
    with open('song_id_to_name.pkl', 'wb') as f:
        pickle.dump(song_id_to_name, f)
    print("Song ID to Name mapping saved to 'song_id_to_name.pkl'.")

print("Training and saving completed.")
