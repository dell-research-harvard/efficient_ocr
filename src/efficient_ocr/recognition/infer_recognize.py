
import numpy as np

END_PUNCTUATION = '.?!,;:"\''

def infer_chars(word_results, recognizer):
    pass

def infer_words(last_char_results, recognizer):
    pass

def infer_last_chars(localizer_results, recognizer):

    # Assemble all the last chars from the results dictionary into a list
    last_chars = []
    no_overlaps_words = []
    for bbox_idx in localizer_results.keys():
        for line_idx in localizer_results[bbox_idx].keys():
            for i, overlaps in enumerate(localizer_results[bbox_idx][line_idx]['overlaps']):
                if len(overlaps) > 0:
                    last_chars.append(localizer_results[bbox_idx][line_idx]['chars'][overlaps[-1]][0])
                else: # If a word has no overlaps
                    pass

            localizer_results[bbox_idx][line_idx]['final_puncs'] = [None] * len(localizer_results[bbox_idx][line_idx]['words'])


    # Get the recognizer results from those chars
    results = recognizer.infer(last_chars)
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

            
