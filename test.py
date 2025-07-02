import os
from dataclasses import dataclass, field
import torchaudio
import torch
from model import EncoderForPretraining
import torchaudio.transforms as T
from collections import defaultdict
from tqdm import tqdm
import glob
from train import Config as ModelConfig

@dataclass
class KNNPipelineConfig:
    k: int = 100
    distance_metric: str = 'inner_product'
    class_info: dict = field(default_factory=lambda: {
        'gw_cbc': 'datasets/gw/cbc',
        'gw_gn': 'datasets/gw/gn',
        'leaves_CEP_LCs': 'datasets/leaves/CEP_LCs',
        'leaves_DSCT_LCs': 'datasets/leaves/DSCT_LCs',
        'leaves_EB_LCs': 'datasets/leaves/EB_LCs',
        'leaves_LPV_LCs': 'datasets/leaves/LPV_LCs',
        'leaves_ROT_LCs': 'datasets/leaves/ROT_LCs',
        'leaves_RR_LCs': 'datasets/leaves/RR_LCs',
        'stead_noise': 'datasets/stead/noise',
        'stead_seism': 'datasets/stead/seism',
        'sleep_class0': 'datasets/sleep/class0',
        'sleep_class1': 'datasets/sleep/class1',
        'sleep_class2': 'datasets/sleep/class2',
        'sleep_class3': 'datasets/sleep/class3',
        'sleep_class4': 'datasets/sleep/class4',
    })
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class KNNPipeline:
    def __init__(self, emb, config: KNNPipelineConfig):
        self.emb = emb.to(config.device)
        self.config = config

        subclass2sample = {}
        k = config.k
        for class_name, class_path in config.class_info.items():
            files = os.listdir(class_path)
            samples = [os.path.join(class_path, f) for f in files if f.endswith('.wav')]
            samples = samples[:k]
            samples = [torchaudio.load(sample)[0] for sample in samples]
            subclass2sample[class_name] = samples

        subclass2emb = defaultdict(list)
        batch_size = 1
        for class_name, samples in subclass2sample.items():
            for i in tqdm(range(0, len(samples), batch_size)):
                batch_samples = samples[i:i + batch_size]
                batch_samples = torch.concat(batch_samples, dim=0).to(self.config.device)
                embeddings = self.batch_embedding(batch_samples)
                subclass2emb[class_name].append(embeddings)
            subclass2emb[class_name] = torch.concat(subclass2emb[class_name], dim=0)
            

        class2emb = {
            'gw': [],
            'leaves': [],
            'stead': [],
            'sleep': [],
        }
        for k, v in subclass2emb.items():
            if k.startswith('gw'):
                class2emb['gw'].append(v)
            elif k.startswith('leaves'):
                class2emb['leaves'].append(v)
            elif k.startswith('stead'):
                class2emb['stead'].append(v)
            elif k.startswith('sleep'):
                class2emb['sleep'].append(v)

        for k, v in class2emb.items():
            class2emb[k] = torch.concat(v, dim=0)
        
        self.subclass2emb = subclass2emb
        self.class2emb = class2emb

    def batch_embedding(self, samples):
        embeddings =  self.emb.encode(samples)
        return embeddings
    
    def infer_knn(self, samples):
        
        samples = samples.to(self.config.device) # shape: (1, time)
        embedding = self.batch_embedding(samples)  # shape: (1, emb_dim)
        embedding = embedding.squeeze(0)  # shape: (emb_dim,)

        # Gather all class embeddings
        class_scores = {}
        for class_name, class_embs in self.class2emb.items():
            # class_embs: (num_samples, emb_dim)
            if self.config.distance_metric == 'inner_product':
                # Higher is more similar
                sims = torch.matmul(class_embs, embedding)
            elif self.config.distance_metric == 'l2':
                # Lower is more similar
                sims = -torch.norm(class_embs - embedding, dim=1)
            else:
                raise ValueError(f"Unknown distance metric: {self.config.distance_metric}")
            
            topk_sims = torch.topk(sims, 100).values
            score = torch.mean(topk_sims).item()
            class_scores[class_name] = score

        # Return the class with the highest score
        pred_class = max(class_scores, key=class_scores.get)
        return pred_class, class_scores


def simple_test(root_dir):
    count = 0
    correct = 0
    label = root_dir.split('/')[-1]
    files = glob.glob(os.path.join(root_dir, '**', '*.wav'), recursive=True)
    for wav_path in tqdm(files):
        if wav_path.endswith('.wav'):
            try:
                sample = torchaudio.load(wav_path)[0]
                predicted_class, scores = model.infer_knn(sample)
                if predicted_class == label:
                    correct += 1
                count += 1
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")
    accuracy = correct / count if count > 0 else 0
    print(f"Accuracy for {label}: {accuracy:.2f} ({correct}/{count})")


if __name__ == "__main__":
    model_config = ModelConfig()
    encoder = EncoderForPretraining(model_config)
    encoder.load_state_dict(torch.load('output/model_final.pth', map_location='cpu'))
    encoder = encoder.to('cuda' if torch.cuda.is_available() else 'cpu')
    model = KNNPipeline(encoder, KNNPipelineConfig())

    simple_test('test_datasets/gw')
    simple_test('test_datasets/stead')
    simple_test('test_datasets/leaves')
    simple_test('test_datasets/sleep')



