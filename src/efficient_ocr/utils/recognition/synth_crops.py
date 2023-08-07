####Create crops given a word list

# Generate words using words from a list (dictionary/place names)

import torchvision.transforms as T
import numpy as np
import os
from tqdm import tqdm

from PIL import ImageOps, Image, ImageFont, ImageDraw
import os
from tqdm import tqdm
import json
from glob import glob
from fontTools.ttLib import TTFont
import multiprocessing as mp
from torch import nn

####Font utils


def word_code_to_string(word_code):

    split_code = word_code.split("_")
    if len(split_code) == 1:
        return chr(int(word_code))
        
    else:
        return "".join([chr(int(char)) for char in split_code])


def string_to_word_code(word_string):
    return "_".join([str(ord(char)) for char in word_string])


def word_crop_dir_to_txt(word_crop_dir, save_path):
    word_crop_list = glob(word_crop_dir+"/*")
    ###Only keep directories
    word_crop_list = [word_crop for word_crop in word_crop_list if os.path.isdir(word_crop)]
    word_crop_list = [word_crop.split('/')[-1] for word_crop in word_crop_list]
    word_crop_list = [word_code_to_string(word_crop) for word_crop in word_crop_list]
    with open(save_path, 'w') as f:
        f.write("\n".join(word_crop_list))
    

def load_chars(path):
    with open(path) as f:
        uni = f.read().split("\n")
    return [u.split("\t") for u in uni]


def get_unicode_chars_font(font_file_path):
    """Get all unicode characters in a font file"""
    font = TTFont(font_file_path, fontNumber=0)

    cmap = font['cmap']

 
    cmap_table = cmap.getBestCmap()

    unicode_chars = [chr(c) for c in cmap_table.keys()]
    return unicode_chars


####Render chars


def render_seg(font_paths, save_path, font_path_id, random_chars_and_spaces, rand_size,ascender_char=True): # You can change the imid into folder id
    # make the folders for this iteration of one font_path_id, you can also iterate over font_path_id inside render_seg

    int_chars=[str(ord(c)) for c in random_chars_and_spaces]
    ###Make the folder for this font_path_id
    folder_name = f"{'_'.join(int_chars)}"
    os.makedirs(os.path.join(save_path,folder_name), exist_ok=True)

    rand_font_path = font_paths[font_path_id] # So we are in one font_path
    digital_font = ImageFont.truetype(rand_font_path, size=rand_size)
    
    if len(random_chars_and_spaces)==1:
        if ascender_char:
            canvas_H=draw_single_char_ascender(random_chars_and_spaces, digital_font, canvas_size=256, padding=0.05)
        else:
            canvas_H=draw_single_char(random_chars_and_spaces, digital_font, canvas_size=256, padding=0.05) ##Size is constant for characters
    else:
        canvas_H=draw_word_from_text(random_chars_and_spaces, digital_font, font_size=rand_size)
    
    font_name = font_paths[font_path_id].split('/')[-1].split('.')[0]
    image_name_H = f"{font_name}-word-{folder_name}.png"

    ##Save an image only if it is > 1 pixel in both dim
    if canvas_H is not None:
        if canvas_H.size[0] > 5 and canvas_H.size[1] > 5:
            canvas_H.save(os.path.join(save_path,folder_name,image_name_H))
    

    return image_name_H,canvas_H


def draw_single_char(ch, font, canvas_size, padding=0.):
    img = Image.new("L", (canvas_size * 4, canvas_size * 4), 0)
    c_w, c_h = img.size
    draw = ImageDraw.Draw(img)
    try:
        draw.text(
            (c_w // 2, c_h // 2), 
            ch, canvas_size, font=font, 
            anchor="mm"
        )
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    # l, u = max(0, l - 5), max(0, u - 5)
    # r, d = min(canvas_size * 2 - 1, r + 5), min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    xdist, ydist = abs(l-r), abs(u-d)
    img = np.array(img)
    img = img[u-int(padding*ydist):d+int(padding*ydist), l-int(padding*xdist):r+int(padding*xdist)]
    img = 255 - img
    img = Image.fromarray(img)
    width, height = img.size
    try:
        img = T.ToTensor()(img)
    except SystemError as e:
        print(e)
        return None
    img = img.unsqueeze(0) 
    pad_len = int(abs(width - height) / 2)  
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    img = img.squeeze(0)
    img = T.ToPILImage()(img)
    img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    return img


def draw_single_char_ascender(ch, font, canvas_size, padding=0.):
    canvas_width, canvas_height = (canvas_size * 5, canvas_size * 5)
    img = Image.new('RGB', (canvas_width, canvas_height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((0,0), ch, (255, 255, 255), font=font)
    bbox = img.getbbox()
    w, h = font.getsize(ch)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    vdist, hdist = y1-y0, x1-x0
    x0, y0, x1, h = x0-(hdist*padding), y0-(vdist*padding), x1+(hdist*padding), h+(vdist*padding)
    uninverted_image = img.crop((x0, 0, x1, h))
    return ImageOps.invert(uninverted_image)


def draw_word_from_text(text,font,font_size):
    """Draw a word given a text string"""
    n=len(text)
    img = Image.new('RGB', (font_size*n*2, font_size*n*2), (0,0,0))
    draw = ImageDraw.Draw(img)
    draw.text((font_size*n,font_size*n), text, (255, 255, 255), font=font,anchor='ms',align='center')

    ##Get bb and crop image with that bb
    # img_copy=img.copy()
    ##Invert a copy of the image
    # img_copy = ImageOps.invert(img_copy)
    if img is not None:
        bbox = img.getbbox()
        if bbox is not None:
            x0,y0,x1,y1 = bbox
            # p = font_size // 25
            # pbbox = (x0-p,y0-p,x1+p,y1+p)

            ##Add some padding
            p = font_size // 25
            pbbox = (x0-p,y0-p,x1+p,y1+p)
            crop = img.crop(pbbox)
            
            crop=ImageOps.invert(crop)


            return crop
        else:
            return None
    else:
        return None
    

def process_word_list(path_to_words,subset_N=None):

    word_list = []
    for wl in path_to_words:
        with open(wl, 'r') as f:
            word_list = f.read().splitlines()
            ## LEGACY
            word_list.extend([word.split(' ')[0] if \
                              len(word.split(' '))==1 else word.split(' ')[1] for word in word_list])

    if subset_N is not None:
        word_list=word_list[:subset_N]

    ###For all words in the word list, generate 3 versions - lowercase, uppercase, title case
    word_list_lower = [word.lower() for word in word_list]
    word_list_upper = [word.upper() for word in word_list]
    # word_list_title = [word.title() for word in word_list]
    word_list_title = [word[0].upper() + word[1:].lower() for word in word_list]

    ###Arrange the word list such that the lower case words are first, then upper case, then title case and the original order is retained
    word_list=[]
    for i in range(0,len(word_list_lower)):
        word_list.append(word_list_lower[i])
        word_list.append(word_list_upper[i])
        word_list.append(word_list_title[i])
    
    word_list=list(set(word_list))

    ###Remove None
    word_list = [word for word in word_list if word is not None]
    ###Remove blanks
    word_list = [word for word in word_list if word != ' ']
    ###Remove empty strings
    word_list = [word for word in word_list if word != '']

    return word_list

 
def render_save_word_list(word_list, font_paths, coverage_dict, save_path,ascender_char=True):
    word_list_covered = []
    for word in word_list:
        if word in word_list_covered:
            continue
        rand_size = np.random.choice(range(70, 133))
        for font_path_id in range(len(font_paths)):

            char_not_covered = False
            for char in word:
                if char not in coverage_dict[font_paths[font_path_id]]:
                    char_not_covered = True
                    continue
            image_name_H, rimg_H = render_seg(font_paths, save_path, font_path_id, word, rand_size,ascender_char=ascender_char)
            word_list_covered.append(word)
    return word_list_covered  


def parallel_render_save_word_list(word_list, font_paths, coverage_dict, save_path, num_processes=4,ascender_char=True):
    
    if num_processes is None:
        num_processes = mp.cpu_count()

    # create a multiprocessing pool with the desired number of processes
    pool = mp.Pool(processes=num_processes)

    # divide the word_list into chunks
    chunk_size = len(word_list) // num_processes
    word_list_chunks = [word_list[i:i+chunk_size] for i in range(0, len(word_list), chunk_size)]

    # execute the function on each chunk in parallel
    results = [pool.apply_async(render_save_word_list, args=(word_list_chunk, font_paths, coverage_dict, save_path,ascender_char)) for word_list_chunk in word_list_chunks]

    # collect the results from each process
    word_list_covered = []
    for result in results:
        if result.get() is not None:
            word_list_covered += result.get()

    # close the pool
    pool.close()
    pool.join()

    return word_list_covered


def render_all_synth_in_parallel(
        save_path:str,
        font_dir:list,
        word_list_path:list, ##This can be characters or words in a txt file separated by "\n",
        ascender_char:bool=True
):
    """
    :param save_path: path to save the images
    :param font_paths: list of font paths
    :param word_list_path: path to a txt file with words separated by "\n"
    """

    ##Make the dir if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    ##All font paths
    font_paths = glob(font_dir+"/*.ttf")
    font_paths = sorted(font_paths)

    char_list   = []
    for font_path in tqdm(font_paths):
        font_path=font_path.replace("\\","/")
        char_list.extend(get_unicode_chars_font(font_path))

    char_list=list(set(char_list))

    coverage_dict = {}
    for font_path in tqdm(font_paths):
        covered_chars = get_unicode_chars_font(font_path)
        covered_chars_kanji_plus = list(set(covered_chars))
        coverage_dict[font_path] = covered_chars_kanji_plus
    with open("/".join(save_path.split("/")[:-1])+"/coverage_dict.json",'w') as f:
        json.dump(coverage_dict,f,ensure_ascii=False)

    # I think the coverage_dict is enough to get the correct chars. 
    ###Keep only those characters in char list that are covered by at least 1 font

    char_list = list(set(char_list))
    char_list = [char for char in char_list if sum([char in coverage_dict[font_path] for font_path in coverage_dict])>=1]
    
    ###Write the wods from a dir to a txt file
    ##GEn symspell words
   
    words_to_generate=process_word_list(path_to_words=word_list_path) 
    
    # covered_symspell=render_save_word_list(symspell_words)
    covered_symspell=parallel_render_save_word_list(
        words_to_generate, font_paths, coverage_dict, save_path, num_processes=None,ascender_char=ascender_char)
    
    ###Now, we need to add the words that are in labels but not in the word list. 
    # print(len(covered_symspell), " images generated")


# if __name__ == '__main__':

#     # Change the path to CGIS
    
#     save_path = '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/test_word_char/images'
#     ##Create folder if not exist
#     # if not os.path.exists(save_path):
#     #     os.makedirs(save_path)
    
#     # Change: please change the font paths to all available CJK font paths. Put everything in here, and also need to SCP these files to CGIS
#     # font_paths = glob("/path/to/data/CJK_fonts/*.ttf")    
#     # font_paths = glob("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/ocr-as-retrieval/english_font_files/*.ttf")
#     # font_paths = glob("/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/ocr-as-retrieval/english_font_files/*.ttf")
#     font_dir = "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/word_level_effocr/extra_font_styles/"
#     ###Remove fonts 10-15
#     text_list_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress/silver_dpd/full_wordlist_effocr.txt"
#     text_list_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/jake_github_repos/ocr-as-retrieval/edgenextSmall_effocrGold_onlySynth_augV3/ref.txt"
#     render_all_in_parallel(save_path, font_dir, text_list_path)