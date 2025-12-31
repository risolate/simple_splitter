from glob import glob
import torch
import torchaudio
from torch.utils.data import Dataset


class hug_musdbhq(Dataset):
    def __init__(self, sub_dataset, duration=7, sample_rate=44100, n_fft = 1024):
        self.sub_dataset = sub_dataset
        self.duration = duration
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.chunks = []
        self._prepare_chunks()

    def _prepare_chunks(self):

        for i in range(len(self.sub_dataset)//5):
            total_samples = (len(self.sub_dataset[5*i]['audio']['bytes'])//4 - 11)

            sr = self.sample_rate

            chunk_samples = int(self.duration * sr)
            num_chunks = total_samples // chunk_samples

            # 전체 청크 추가
            for j in range(num_chunks):
                start_sample = j * chunk_samples
                self.chunks.append((i, start_sample, chunk_samples))

            # 마지막 남은 부분 추가
            remaining_samples = total_samples % chunk_samples
            if remaining_samples > 0:
                start_sample = num_chunks * chunk_samples
                self.chunks.append((i, start_sample, remaining_samples))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):

        audio_idx, start_sample, num_samples = self.chunks[idx]

        frame_offset = start_sample
        num_frames = num_samples

        # 오디오의 특정 부분만 로드
        waveform, sr = torchaudio.load(
            self.sub_dataset[5*audio_idx]['audio']['bytes'],
            frame_offset=frame_offset,
            num_frames=num_frames
        )
        vc_waveform, vc_sr = torchaudio.load(
            self.sub_dataset[5*audio_idx+4]['audio']['bytes'],
            frame_offset=frame_offset,
            num_frames=num_frames
        )

        # 청크 길이에 맞게 패딩 추가
        target_len_samples = int(self.duration * sr)
        current_len_samples = waveform.shape[1]

        if current_len_samples < target_len_samples:
            padding_len = target_len_samples - current_len_samples
            padding = torch.zeros((waveform.shape[0], padding_len))
            waveform = torch.cat([waveform, padding], dim=1)
            vc_waveform = torch.cat([vc_waveform,padding],dim=1)

        waveform = waveform[:, :target_len_samples]
        vc_waveform = vc_waveform[:, :target_len_samples]

        #모노로 전환
        waveform = waveform.mean(dim=0)
        vc_waveform = vc_waveform.mean(dim=0)

        spec = torch.functional.stft(
                                    waveform,
                                    window=torch.hann_window(self.n_fft).to(waveform),
                                    n_fft = self.n_fft, 
                                    return_complex = True,
                                    )      # fq * T * 2
        spec = torch.view_as_real(spec).permute(2,0,1)
        labels = vc_waveform
        item = {'input_ids': spec , 'labels': labels}

        return item