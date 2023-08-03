
def check_any_overlap(bbox_1, bbox_2):
    """Check if two bboxes overlap, we do this by checking all four corners of bbox_1 against bbox_2"""

    x1, y1, x2, y2 = bbox_1
    x3, y3, x4, y4 = bbox_2

    return x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3

def check_hoi_overlap(bbox_1, bbox_2):
    """ Check if two bboxes overlap horizontally, we do this by checking the right and left sides of bbox_1 against bbox_2"""
    
    x1, y1, x2, y2 = bbox_1
    x3, y3, x4, y4 = bbox_2

    return x1 < x4 and x2 > x3

def find_overlaps(sorted_chars, sorted_words):
    # For each word, find all chars it overlaps with:
    word_char_idx = [[] for _ in range(len(sorted_words))]
    word_idx = 0
    for char_idx, char_bbox in enumerate(sorted_chars):
        if word_idx >= len(sorted_words):
            break
        
        orig_idx = word_idx
        while word_idx < len(sorted_words) and not check_any_overlap(sorted_words[word_idx], char_bbox):
            word_idx += 1
        
        # If the detected character is oddly positioned, like squished into the top of the screen or similar, we will
        # not find an overlapping word and will need to reset.
        if word_idx >= len(sorted_words):
            word_idx = orig_idx
        else:
            word_char_idx[word_idx].append(char_idx)

    return word_char_idx

def en_preprocess(bboxes_char, bboxes_word, vertical=False):

    sorted_bboxes_char = sorted(bboxes_char, key=lambda x: x[1] if vertical else x[0])
    sorted_bboxes_word = sorted(bboxes_word, key=lambda x: x[1] if vertical else x[0])

    # Find all overlaps between chars and words
    word_char_idx = find_overlaps(sorted_bboxes_char, sorted_bboxes_word)
        
    # # For each word, find all chars it overlaps with
    # word_char_idx = []
    # for word_bbox in sorted_bboxes_word:
    #     word_char_idx.append([])
    #     for char_idx, char_bbox in enumerate(sorted_bboxes_char):
    #         if check_any_overlap(word_bbox, char_bbox):
    #             word_char_idx[-1].append(char_idx)

    # If there are no overlapping chars for a word, append the word bounding box to the list of chars as a char
    redo_list, to_remove = False, []
    for i, word_bbox in enumerate(sorted_bboxes_word):
        if len(word_char_idx[i]) == 0:
            remove = False
            for j, comp_word_bbox in enumerate(sorted_bboxes_word):
                if i != j and check_hoi_overlap(word_bbox, comp_word_bbox):
                    remove = True
                    to_remove.append(i)
                    break

            if not remove:
                sorted_bboxes_char.append(word_bbox)
                redo_list = True
    
    for i in sorted(to_remove, reverse=True):
        del sorted_bboxes_word[i]
        del word_char_idx[i]


    # If we found a word with no overlapping chars, we now need to resort the char list and recreate the word_char_idx list
    if redo_list:
        # Resort the sorted_bboxes_char list and adjust the word_char_idx list accordingly
        sorted_bboxes_char = sorted(sorted_bboxes_char, key=lambda x: x[1] if vertical else x[0])
        word_char_idx = find_overlaps(sorted_bboxes_char, sorted_bboxes_word)

    if any([len(w) == 0 for w in word_char_idx]):
        print('Error: word_char_idx contains a list with no elements')
        print(word_char_idx)
        print(sorted_bboxes_char)
        print(sorted_bboxes_word)
        print(bboxes_char)
        print(bboxes_word)
        print(redo_list)
        raise ValueError('word_char_idx contains a list with no elements')
    # Return the lists of chars, words, and overlaps
    return sorted_bboxes_char, sorted_bboxes_word, word_char_idx
