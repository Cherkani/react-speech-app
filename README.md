py -3.9 -m venv myenv
pip install -r requirements.txt
pip install numpy<2

#change featurable file in packages

 with open(weights_fn)
to 
 with open(weights_fn,encoding='utf-8')
#change 



def get_resulting_string(mapped_indices: np.array, words_estimated: list, words_real: list) -> list:
    mapped_words = []
    mapped_words_indices = []
    WORD_NOT_FOUND_TOKEN = '-'
    number_of_real_words = len(words_real)
    for word_idx in range(number_of_real_words):
        position_of_real_word_indices = np.where(
            mapped_indices == word_idx)[0].astype(int)
        if len(position_of_real_word_indices) == 0:
            mapped_words.append(WORD_NOT_FOUND_TOKEN)
            mapped_words_indices.append(-1)
            continue
        if len(position_of_real_word_indices) == 1:
            mapped_words.append(
                words_estimated[position_of_real_word_indices[0]])
            mapped_words_indices.append(position_of_real_word_indices[0])
            continue
        # Check which index gives the lowest error
        if len(position_of_real_word_indices) > 1:
            error = 99999
            best_possible_combination = ''
            best_possible_idx = -1
            for single_word_idx in position_of_real_word_indices:
                idx_above_word = single_word_idx >= len(words_estimated)
                if idx_above_word:
                    continue
                error_word = WordMetrics.edit_distance_python(
                    words_estimated[single_word_idx], words_real[word_idx])
                if error_word < error:
                    error = error_word*1
                    best_possible_combination = words_estimated[single_word_idx]
                    best_possible_idx = single_word_idx
            mapped_words.append(best_possible_combination)
            mapped_words_indices.append(best_possible_idx)
            continue
    return mapped_words, mapped_words_indices
