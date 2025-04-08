import torch
from torch.nn.utils.rnn import pad_sequence
from espnet2.tasks.ssl import SSLTask
import soundfile as sf
import numpy as np

from pydub import AudioSegment
from io import BytesIO




device = "cuda" if torch.cuda.is_available() else "cpu"

xeus_model, xeus_train_args =  SSLTask.build_model_from_file(
    None,
    "./XEUS/model/xeus_checkpoint.pth",
    device
)


# # Load audio with soundfile
wavs, sample_rate = sf.read("./audio/movie_audio_segments_wav/kae/kae_55_62.wav")

# # # Convert to mono if stereo
# # if len(wavs.shape) > 1 and wavs.shape[1] > 1:
# #     wavs = np.mean(wavs, axis=1)

# Convert to torch tensor with shape [batch, time]
wav_lengths = torch.LongTensor([len(wav) for wav in [wavs]]).to(device)
wavs = pad_sequence(torch.Tensor([wavs]), batch_first=True).to(device) 

# we recommend use_mask=True during fine-tuning
feats = xeus_model.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0][-1] # take the output of the last layer -> batch_size x seq_len x hdim

print(f"Wav shape: {wavs.shape}")
print(f"Sample Rate: {sample_rate}")
print(f"Wav Lengths: {wav_lengths}")
print(f"Features shape: {feats}")