import os
import argparse
import librosa
import soundfile as sf
import numpy as np

def create_silence_file(reference_audio_path, output_path, sample_rate=8000):
    """音声ファイルと同じ長さの無音ファイルを生成する"""
    try:
        duration = librosa.get_duration(path=reference_audio_path)
        silence = np.zeros(int(duration * sample_rate))
        sf.write(output_path, silence, sample_rate)
        return True
    except Exception as e:
        print(f"Error creating silence file for {reference_audio_path}: {e}")
        return False

def create_silence_files_for_librimix(librimix_base):
    """LibriMixデータセット用の無音ファイルを一括作成する"""
    
    # 無音ファイルを格納するディレクトリを作成
    silence_dirs = [
        f'{librimix_base}/train-100/silence/s1',
        f'{librimix_base}/train-100/silence/s2',
        f'{librimix_base}/test/silence/s1',
        f'{librimix_base}/test/silence/s2',
        f'{librimix_base}/dev/silence/s1',
        f'{librimix_base}/dev/silence/s2'
    ]
    for silence_dir in silence_dirs:
        os.makedirs(silence_dir, exist_ok=True)

    # (s1, s2, s1_silence, s2_silence)
    datasets = [
        ('train-100', f'{librimix_base}/train-100/s1', f'{librimix_base}/train-100/s2', silence_dirs[0], silence_dirs[1]),
        ('test', f'{librimix_base}/test/s1', f'{librimix_base}/test/s2', silence_dirs[2], silence_dirs[3]),
        ('dev', f'{librimix_base}/dev/s1', f'{librimix_base}/dev/s2', silence_dirs[4], silence_dirs[5])
    ]

    for dataset_name, s1_dir, s2_dir, s1_silence_dir, s2_silence_dir in datasets:
        print(f"Generating silence files for {dataset_name}...")
        
        # 対応する無音ファイル生成
        # s1
        for root, dirs, files in os.walk(s1_dir):
            files.sort()
            for file in files:
                if file.endswith('.wav'):
                    silence_file_path = os.path.join(s1_silence_dir, file)
                    if not os.path.exists(silence_file_path):
                        create_silence_file(os.path.join(root, file), silence_file_path)
        # s2
        for root, dirs, files in os.walk(s2_dir):
            files.sort()
            for file in files:
                if file.endswith('.wav'):
                    silence_file_path = os.path.join(s2_silence_dir, file)
                    if not os.path.exists(silence_file_path):
                        create_silence_file(os.path.join(root, file), silence_file_path)
    
    print("Silence files generation completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create silence files for LibriMix dataset')
    parser.add_argument('--base', type=str, required=True, help='Base path for LibriMix dataset')
    args = parser.parse_args()
    
    create_silence_files_for_librimix(args.base)
