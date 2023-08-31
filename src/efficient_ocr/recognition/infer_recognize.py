

END_PUNCTUATION = '.?!,;:"\''


def infer_chars(word_results, recognizer, char_only=False):
    # Assemble all the char crops from the results dictionary into a list
    char_crops = []
    for bbox_idx in word_results.keys():
        if not char_only:
            for line_idx in word_results[bbox_idx].keys():
                for i, p in enumerate(word_results[bbox_idx][line_idx]['word_preds']):
                    if p is not None:
                        continue
                    else:
                        for overlap in word_results[bbox_idx][line_idx]['overlaps'][i]:
                            char_crops.append(word_results[bbox_idx][line_idx]['chars'][overlap][0])
        else:
            for line_idx in word_results[bbox_idx].keys():

                for i, c in enumerate(word_results[bbox_idx][line_idx]['chars']):
                    char_crops.append(word_results[bbox_idx][line_idx]['chars'][i][0])

    # Get the recognizer results from those chars
    results = recognizer.run(char_crops)
    assert len(results) == len(char_crops)

    # Assemble the results back into the word_results dictionary
    results_idx = 0
    for bbox_idx in word_results.keys():
        for line_idx in word_results[bbox_idx].keys():
            if not char_only:
                for i, p in enumerate(word_results[bbox_idx][line_idx]['word_preds']):
                    if p is not None:
                        continue
                    else:
                        word = ''
                        for _ in range(len(word_results[bbox_idx][line_idx]['overlaps'][i])):
                            word += results[results_idx]
                            results_idx += 1
                        word_results[bbox_idx][line_idx]['word_preds'][i] = word
            else:
                for i in range(len(word_results[bbox_idx][line_idx]['chars'])):
                    word_results[bbox_idx][line_idx]['word_preds'][i] = results[results_idx]
                    results_idx += 1

            
    
    # Add final punctuation to the end of each word
    if not char_only:
        for bbox_idx in word_results.keys():
            for line_idx in word_results[bbox_idx].keys():
                for i, p in enumerate(word_results[bbox_idx][line_idx]['final_puncs']):
                    if p is not None:
                        word_results[bbox_idx][line_idx]['word_preds'][i] += p
                
                if word_results[bbox_idx][line_idx]['para_end']:
                    if len(word_results[bbox_idx][line_idx]['word_preds']) > 0:
                        word_results[bbox_idx][line_idx]['word_preds'][-1] += '\n'
                    else:
                        word_results[bbox_idx][line_idx]['word_preds'].append('\n')

    return word_results


def infer_words(last_char_results, recognizer, threshold = 0.83):
    # Assemble all the word crops from the results dictionary into a list
    word_crops = []
    for bbox_idx in last_char_results.keys():
        for line_idx in last_char_results[bbox_idx].keys():
            word_crops.extend([last_char_results[bbox_idx][line_idx]['words'][i][0] for i in range(len(last_char_results[bbox_idx][line_idx]['words']))])
            last_char_results[bbox_idx][line_idx]['word_preds'] = [None] * len(last_char_results[bbox_idx][line_idx]['words'])

    # Get the recognizer results from those chars
    results = recognizer.run(word_crops, cutoff = threshold)
    assert len(results) == len(word_crops)

    # Assemble the results back into the last_char_results dictionary
    results_idx = 0
    for bbox_idx in last_char_results.keys():
        for line_idx in last_char_results[bbox_idx].keys():
            for i in range(len(last_char_results[bbox_idx][line_idx]['word_preds'])):
                last_char_results[bbox_idx][line_idx]['word_preds'][i] = results[results_idx]
                results_idx += 1

    return last_char_results


def infer_last_chars(localizer_results, recognizer):

    # Assemble all the last chars from the results dictionary into a list
    last_chars = []

    for bbox_idx in localizer_results.keys():
        for line_idx in localizer_results[bbox_idx].keys():
            for i, overlaps in enumerate(localizer_results[bbox_idx][line_idx]['overlaps']):
                if len(overlaps) > 0:
                    last_chars.append(localizer_results[bbox_idx][line_idx]['chars'][overlaps[-1]][0])
                else: # If a word has no overlaps
                    pass

            localizer_results[bbox_idx][line_idx]['final_puncs'] = [None] * len(localizer_results[bbox_idx][line_idx]['words'])


    # Get the recognizer results from those chars
    results = recognizer.run(last_chars)
    assert len(results) == len(last_chars)

    # Assemble the results back into the localizer_results dictionary
    results_idx = 0
    for bbox_idx in localizer_results.keys():
        for line_idx in localizer_results[bbox_idx].keys():
            for i, overlaps in enumerate(localizer_results[bbox_idx][line_idx]['overlaps']):
                if len(overlaps) > 0:
                    if results[results_idx] in END_PUNCTUATION:
                        localizer_results[bbox_idx][line_idx]['final_puncs'][i] = results[results_idx] # Set the final punctuation in the list
                        # Get the difference between right edges of word and character boxes
                        to_remove = localizer_results[bbox_idx][line_idx]['words'][i][1][1] - localizer_results[bbox_idx][line_idx]['chars'][overlaps[-1]][1][1] 
                        # Adjust the word bounding box to reflect the removal of the last character
                        if to_remove > 0:
                            localizer_results[bbox_idx][line_idx]['words'][i][0] = localizer_results[bbox_idx][line_idx]['words'][i][0][:, :(-1 * to_remove)] # Remove the left edge of the word image
                        # Remove the last character from the list of overlapping characters
                        localizer_results[bbox_idx][line_idx]['overlaps'][i] = overlaps[:-1] # Remove the last character from the overlaps list 
                        
                    
                    results_idx += 1
                
                else: # If a word has no overlaps and thus no last character
                    pass

    return localizer_results

