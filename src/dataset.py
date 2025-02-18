# dataset.py - Handles dataset loading and preprocessing
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchaudio
import os
import cv2

# Load Audio Features (Lip-Syncing Model)
def extract_audio_features(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
    return mel_spectrogram.mean(dim=0)  # Average channels

# Dataset Loader for Video Frames
class TalkingHeadDataset(Dataset):
    def __init__(self, video_folder, audio_folder):
        self.video_frames = sorted(os.listdir(video_folder))
        self.audio_files = sorted(os.listdir(audio_folder))
        self.video_folder = video_folder
        self.audio_folder = audio_folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.video_frames)
    
    def __getitem__(self, idx):
        frame_path = os.path.join(self.video_folder, self.video_frames[idx])

        # Ensure we don’t exceed available audio files
        audio_idx = min(idx, len(self.audio_files) - 1)
        audio_path = os.path.join(self.audio_folder, self.audio_files[audio_idx])

        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transform(frame)

        audio_features = extract_audio_features(audio_path)
        return frame, audio_features

