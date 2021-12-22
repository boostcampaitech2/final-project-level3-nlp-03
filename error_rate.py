import jiwer

def wer(gt, inferenced):  # (I+D+S)/N*100 (%), N is # of words in gt
    I = 0  # # of insertion
    D = 0  # # of deletion
    S = 0  # # of substitution
    # ref_words = ref_setence.split()
    # input_words = input_sentence.split()

    return jiwer.wer(gt, inferenced)

print(wer('How are you today John', 'How you a today Johnes'))
print(wer('안녕 나는 송 .이현이다', '안녕 나 송이현 .이다'))
print(wer('안녕 나는 송이현 이다 캬캬캬', '안녕 송이현 이다 캬캬'))
