import os
from pathlib import Path

import librosa

from audio_zen.acoustics.feature import load_wav
from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.utils import basename
from audio_zen.utils import expand_path

class VoiceBankDataset(BaseDataset):
    def __init__(
        self,
        dataset_dir_list,
        clean_dataset,
        noise_dataset,
        sr,
    ):
        """
        Construct DNS validation set

        synthetic/
            with_reverb/
                noisy/
                clean_y/
            no_reverb/
                noisy/
                clean_y/
        """
        super(VoiceBankDataset, self).__init__()
        
        self.clean_dataset_list = [
            line.rstrip("\n") for line in open(expand_path(clean_dataset), "r")
        ]
        
        self.noisy_dataset_list = [
            line.rstrip("\n") for line in open(expand_path(noise_dataset), "r")
        ]

        self.sr = sr

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        """
        use the absolute path of the noisy speech to find the corresponding clean speech.

        Notes
            with_reverb and no_reverb dirs have same-named files.
            If we use `basename`, the problem will be raised (cover) in visualization.

        Returns:
            noisy: [waveform...], clean: [waveform...], type: [reverb|no_reverb] + name
        """
        noisy_file_path = self.noisy_dataset_list[item]
        clean_file_path = self.clean_dataset_list[item]

        noisy_filename, _ = basename(noisy_file_path)

        noisy = load_wav(os.path.abspath(os.path.expanduser(noisy_file_path)), sr=self.sr)
        clean = load_wav(os.path.abspath(os.path.expanduser(clean_file_path)), sr=self.sr)

        return noisy, clean

        
        return noisy, clean
