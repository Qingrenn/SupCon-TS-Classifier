import os
import torch
from torch.utils.data import Dataset
import torchaudio
import concurrent.futures
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_subclass(i_j_subcls_path):
    i, j, subcls_path = i_j_subcls_path
    fcls = os.path.basename(os.path.dirname(subcls_path))
    subcls = os.path.basename(subcls_path)
    
    samples = []
    wav_files = [fname for fname in os.listdir(subcls_path) if fname.endswith('.wav')]
    for fname in tqdm(wav_files, desc=f"{fcls}/{subcls}", position=j):
        path = os.path.join(subcls_path, fname)
        samples.append((path, i, j))

    return samples


def process_entry(entry_path, i):
    if entry_path.endswith('.wav'):
        return (entry_path, i, -1)


def parallel_process_subfiles(subfiles_or_dirs, i, max_workers=8):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_entry, path, i) for path in subfiles_or_dirs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(future.result())
    return results


class FolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.class_to_idx = {}
        self.subclass_to_idx = {}
        self.transform = transform
        self._find_samples(root_dir)


    def _find_samples(self, root_dir):
        classes = sorted(os.listdir(root_dir))
        
        for i, fcls in enumerate(classes):
            print(f"Processing class: {fcls}")

            cls_path = os.path.join(root_dir, fcls)
            if not os.path.isdir(cls_path):
                continue
            self.class_to_idx[fcls] = i
            
            subfiles_or_dirs = map(lambda sub: os.path.join(cls_path, sub), os.listdir(cls_path))
            subfiles_or_dirs = list(subfiles_or_dirs)
            if len(subfiles_or_dirs) > 1e5:
                print(f"Warning: {fcls} has more than 100,000 files.")
                subfiles_or_dirs = subfiles_or_dirs[:100000]

            if os.path.isdir(subfiles_or_dirs[0]):
                subclasses = subfiles_or_dirs
            else:
                samples = parallel_process_subfiles(subfiles_or_dirs, i)
                self.samples.extend(samples)
                subclasses = []
            
            inputs = []
            for j, subcls_path in enumerate(subclasses):
                subcls = subcls_path.split('/')[-1]
                self.subclass_to_idx[(fcls, subcls)] = (i, j)
                inputs.append((i, j, subcls_path))
                
            results = process_map(
                process_subclass,
                inputs,
                max_workers=os.cpu_count() or 1,
                desc=f"Processing subclasses in {fcls}"
            )

            for samples in results:
                self.samples.extend(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_idx, subclass_idx = self.samples[idx]
        waveform, sample_rate = torchaudio.load(path)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        return {
            'inputs': waveform.squeeze(0),  # Remove channel dimension if mono
            'class_idx': class_idx,
            'subclass_idx': subclass_idx
        }