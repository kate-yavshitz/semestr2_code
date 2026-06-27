import torchaudio
import argparse
from pyannote.audio import Model
from pyannote.audio import Inference
import pandas as pd
from tqdm import tqdm


def read_audio_from_table(df, index):
    audio_name = df.iloc[index]['name']
    audio_path = df.iloc[index]['path']
    audio, sr = torchaudio.load(audio_path)
    return audio_name, audio


def main(audio_folder):
    # Initialize model
    model = Model.from_pretrained("pyannote/brouhaha",
                                  use_auth_token="",
                                  local_files_only=False)
    inference = Inference(model)

    # Load data
    df = pd.read_csv(f'{audio_folder}/all.csv', sep='\t')

    # Process files
    for i in tqdm(range(len(df))):
        # Get name and path from df
        name = df.iloc[i]['name']
        path = df.iloc[i]['path']
        
        # Run inference on the file path
        output = inference(path)
        print(f"File: {name}\n\n")
        # iterate over each frame
        for frame, (vad, snr, c50) in output:
            t = frame.middle
            print(f"{t:8.3f} vad={100*vad:.0f}% snr={snr:.0f} c50={c50:.0f}")        


# Создаём парсер
parser = argparse.ArgumentParser(description="Передадим папку с записями")

# Добавляем аргументы
parser.add_argument("folder_path", help="Путь к папке с записями")           

args = parser.parse_args()
main(args.folder_path)
