# stt-en

### Install nemo
```
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']==1.1.0
```

### Install LM decoding
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
cd ..
```

### Download pretrained LM model
```
wget https://storage.googleapis.com/mailong25/stt_en_conformer_ctc_large.bin
```

### Infernece
```
# Init model

import os
import nemo.collections.asr as nemo_asr
import ctcdecode
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

ASR_MODEL_NAME = "stt_en_conformer_ctc_large"
LM_MODEL_PATH = '/path/to/pretrained LM model.bin'

asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
vocab = asr_model.decoder.vocabulary
asr_model.eval()
vocab.append('_')
beam_search_lm = ctcdecode.CTCBeamDecoder(
    [chr(idx + 100) for idx in range(len(vocab))],
    model_path=LM_MODEL_PATH,
    alpha=0.5,
    beta=0.5,
    beam_width=24,
    num_processes=max(os.cpu_count(), 1),
    blank_id=len(vocab) - 1,
    cutoff_prob=1.0,
    cutoff_top_n=24,
    log_probs_input=False)
    
def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

def transcribe(asr_model, lm_model, file):
    with torch.no_grad():
        logits = asr_model.transcribe(file, batch_size = 12, logprobs=True)

    probs   = [torch.FloatTensor(softmax(logit)) for logit in logits]
    seq_len = torch.LongTensor([p.shape[0] for p in probs])
    probs = pad_sequence(probs, batch_first = True)
    beam_results, beam_scores, timesteps, out_lens = lm_model.decode(probs, seq_len)
    
    results = []
    for beam_cur,timesteps_cur,lens_cur,probs_cur in zip(beam_results, timesteps, out_lens, probs):
        hypos = []
        confs = []
        for i in range(0,len(lens_cur)):
            if lens_cur[i] == 0:
                hypos.append('')
                confs.append('0.0')
            else:
                hypo_confs = []
                hypos.append(asr_model.tokenizer.ids_to_text(beam_cur[i][:lens_cur[i]].cpu().numpy()))
                for j in range(0,lens_cur[i]):
                    hypo_confs.append(probs_cur[timesteps_cur[i][j]][beam_cur[i][j]])

                hypo_confs = [str(round(float(conf),4)) for conf in hypo_confs]
                confs.append('\t'.join(hypo_confs))
        
        res_one = list(zip(hypos,confs))
        #res_one = [r[0] + '\t' + r[1] for r in res_one]
        res_one = [r[0] for r in res_one][0]
        results.append(res_one)
    
    return results

# Call the inference function
transcripts = transcribe(asr_model, beam_search_lm, ['path/to/audio1.wav,path/to/audio2.wav'])
```
