import os
import argparse

parser = argparse.ArgumentParser(description='Create SCP files for LibriMix dataset')
parser.add_argument('--base', type=str, required=True, help='Base path for LibriMix dataset')
args = parser.parse_args()

LIBRIMIX_BASE = args.base

train_mix_scp = 'scp_ss_8k_libri12/tr_mix.scp'
train_s1_scp = 'scp_ss_8k_libri12/tr_s1.scp'
train_s2_scp = 'scp_ss_8k_libri12/tr_s2.scp'

train_mix = f'{LIBRIMIX_BASE}/train-100/mix_clean'
train_s1 = f'{LIBRIMIX_BASE}/train-100/s1'
train_s2 = f'{LIBRIMIX_BASE}/train-100/s2'
train_s1_silence = f'{LIBRIMIX_BASE}/train-100/silence/s1'
train_s2_silence = f'{LIBRIMIX_BASE}/train-100/silence/s2'

only_suffix = '_only.wav'  # 単話者音声であることを示す末尾の文字列

tr_mix = open(train_mix_scp, 'w')
for root, dirs, files in os.walk(train_mix):
    files.sort()
    for file in files:
        tr_mix.write(file + ' ' + root + '/' + file)
        tr_mix.write('\n')

# 単話者音声をmixに追加
# s1
for root, dirs, files in os.walk(train_s1):
    files.sort()
    for file in files:
        tr_mix.write(file.replace('.wav', '_s1' + only_suffix) + ' ' + root + '/' + file)
        tr_mix.write('\n')

# s2
for root, dirs, files in os.walk(train_s2):
    files.sort()
    for file in files:
        tr_mix.write(file.replace('.wav', '_s2' + only_suffix) + ' ' + root + '/' + file)
        tr_mix.write('\n')
tr_mix.close()

tr_s1 = open(train_s1_scp, 'w')
for root, dirs, files in os.walk(train_s1):
    files.sort()
    for file in files:
        tr_s1.write(file + ' ' + root + '/' + file)
        tr_s1.write('\n')

# 単話者としてのs1を追加
for root, dirs, files in os.walk(train_s1):
    files.sort()
    for file in files:
        tr_s1.write(file.replace('.wav', '_s1' + only_suffix) + ' ' + root + '/' + file)
        tr_s1.write('\n')

# s2に対応する無音ファイルを追加
for root, dirs, files in os.walk(train_s2_silence):
    files.sort()
    for file in files:
        tr_s1.write(file.replace('.wav', '_s2' + only_suffix) + ' ' + root + '/' + file)
        tr_s1.write('\n')
tr_s1.close()

tr_s2 = open(train_s2_scp, 'w')
for root, dirs, files in os.walk(train_s2):
    files.sort()
    for file in files:
        tr_s2.write(file + " " + root + '/' + file)
        tr_s2.write('\n')

# s1に対応する無音ファイルを追加
for root, dirs, files in os.walk(train_s1_silence):
    files.sort()
    for file in files:
        tr_s2.write(file.replace('.wav', '_s1' + only_suffix) + ' ' + root + '/' + file)
        tr_s2.write('\n')

# 単話者音声としてのs2を追加
for root, dirs, files in os.walk(train_s2):
    files.sort()
    for file in files:
        tr_s2.write(file.replace('.wav', '_s2' + only_suffix) + ' ' + root + '/' + file)
        tr_s2.write('\n')
tr_s2.close()

test_mix_scp = 'scp_ss_8k_libri12/tt_mix.scp'
test_s1_scp = 'scp_ss_8k_libri12/tt_s1.scp'
test_s2_scp = 'scp_ss_8k_libri12/tt_s2.scp'

test_mix = f'{LIBRIMIX_BASE}/test/mix_clean'
test_s1 = f'{LIBRIMIX_BASE}/test/s1'
test_s2 = f'{LIBRIMIX_BASE}/test/s2'
test_s1_silence = f'{LIBRIMIX_BASE}/test/silence/s1'
test_s2_silence = f'{LIBRIMIX_BASE}/test/silence/s2'

tt_mix = open(test_mix_scp, 'w')
for root, dirs, files in os.walk(test_mix):
    files.sort()
    for file in files:
        tt_mix.write(file + " " + root + '/' + file)
        tt_mix.write('\n')

# 単話者音声をmixに追加
# s1
for root, dirs, files in os.walk(test_s1):
    files.sort()
    for file in files:
        tt_mix.write(file.replace('.wav', '_s1' + only_suffix) + ' ' + root + '/' + file)
        tt_mix.write('\n')

# s2
for root, dirs, files in os.walk(test_s2):
    files.sort()
    for file in files:
        tt_mix.write(file.replace('.wav', '_s2' + only_suffix) + ' ' + root + '/' + file)
        tt_mix.write('\n')
tt_mix.close()

tt_s1 = open(test_s1_scp, 'w')
for root, dirs, files in os.walk(test_s1):
    files.sort()
    for file in files:
        tt_s1.write(file + ' ' + root + '/' + file)
        tt_s1.write('\n')

# 単話者としてのs1を追加
for root, dirs, files in os.walk(test_s1):
    files.sort()
    for file in files:
        tt_s1.write(file.replace('.wav', '_s1' + only_suffix) + ' ' + root + '/' + file)
        tt_s1.write('\n')

# s2に対応する無音ファイルを追加
for root, dirs, files in os.walk(test_s2_silence):
    files.sort()
    for file in files:
        tt_s1.write(file.replace('.wav', '_s2' + only_suffix) + ' ' + root + '/' + file)
        tt_s1.write('\n')
tt_s1.close()

tt_s2 = open(test_s2_scp, 'w')
for root, dirs, files in os.walk(test_s2):
    files.sort()
    for file in files:
        tt_s2.write(file + ' ' + root + '/' + file)
        tt_s2.write('\n')

# s1に対応する無音ファイルを追加
for root, dirs, files in os.walk(test_s1_silence):
    files.sort()
    for file in files:
        tt_s2.write(file.replace('.wav', '_s1' + only_suffix) + ' ' + root + '/' + file)
        tt_s2.write('\n')

# 単話者音声としてのs2を追加
for root, dirs, files in os.walk(test_s2):
    files.sort()
    for file in files:
        tt_s2.write(file.replace('.wav', '_s2' + only_suffix) + ' ' + root + '/' + file)
        tt_s2.write('\n')
tt_s2.close()

cv_mix_scp = 'scp_ss_8k_libri12/cv_mix.scp'
cv_s1_scp = 'scp_ss_8k_libri12/cv_s1.scp'
cv_s2_scp = 'scp_ss_8k_libri12/cv_s2.scp'

cv_mix = f'{LIBRIMIX_BASE}/dev/mix_clean'
cv_s1 = f'{LIBRIMIX_BASE}/dev/s1'
cv_s2 = f'{LIBRIMIX_BASE}/dev/s2'
cv_s1_silence = f'{LIBRIMIX_BASE}/dev/silence/s1'
cv_s2_silence = f'{LIBRIMIX_BASE}/dev/silence/s2'

cv_mix_file = open(cv_mix_scp, 'w')
for root, dirs, files in os.walk(cv_mix):
    files.sort()
    for file in files:
        cv_mix_file.write(file + ' ' + root + '/' + file)
        cv_mix_file.write('\n')

# 単話者音声をmixに追加
# s1
for root, dirs, files in os.walk(cv_s1):
    files.sort()
    for file in files:
        cv_mix_file.write(file.replace('.wav', '_s1' + only_suffix) + ' ' + root + '/' + file)
        cv_mix_file.write('\n')

# s2
for root, dirs, files in os.walk(cv_s2):
    files.sort()
    for file in files:
        cv_mix_file.write(file.replace('.wav', '_s2' + only_suffix) + ' ' + root + '/' + file)
        cv_mix_file.write('\n')
cv_mix_file.close()

cv_s1_file = open(cv_s1_scp, 'w')
for root, dirs, files in os.walk(cv_s1):
    files.sort()
    for file in files:
        cv_s1_file.write(file + ' ' + root + '/' + file)
        cv_s1_file.write('\n')

# 単話者としてのs1を追加
for root, dirs, files in os.walk(cv_s1):
    files.sort()
    for file in files:
        cv_s1_file.write(file.replace('.wav', '_s1' + only_suffix) + ' ' + root + '/' + file)
        cv_s1_file.write('\n')

# s2に対応する無音ファイルを追加
for root, dirs, files in os.walk(cv_s2_silence):
    files.sort()
    for file in files:
        cv_s1_file.write(file.replace('.wav', '_s2' + only_suffix) + ' ' + root + '/' + file)
        cv_s1_file.write('\n')
cv_s1_file.close()

cv_s2_file = open(cv_s2_scp, 'w')
for root, dirs, files in os.walk(cv_s2):
    files.sort()
    for file in files:
        cv_s2_file.write(file + ' ' + root + '/' + file)
        cv_s2_file.write('\n')

# s1に対応する無音ファイルを追加
for root, dirs, files in os.walk(cv_s1_silence):
    files.sort()
    for file in files:
        cv_s2_file.write(file.replace('.wav', '_s1' + only_suffix) + ' ' + root + '/' + file)
        cv_s2_file.write('\n')

# 単話者音声としてのs2を追加
for root, dirs, files in os.walk(cv_s2):
    files.sort()
    for file in files:
        cv_s2_file.write(file.replace('.wav', '_s2' + only_suffix) + ' ' + root + '/' + file)
        cv_s2_file.write('\n')
cv_s2_file.close()
