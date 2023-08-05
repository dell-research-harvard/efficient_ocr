import torch
import torch.nn as nn
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel, FaissKNN
import logging
import faiss
import os
from torchvision import transforms as T
import numpy as np
logging.getLogger().setLevel(logging.INFO)

from torch.optim import AdamW

import wandb
import argparse
from collections import defaultdict

from models.encoders import *
from datasets.recognizer_datasets import * # make sure Huggingface datasets is not installed...
from utils.datasets_utils import INV_NORMALIZE
from tqdm import tqdm
from utils.custom_schedulers import CosineAnnealingDecWarmRestarts
from synth_crops import render_all_synth_in_parallel

def infer_hardneg_dataset(query_dataset, ref_dataset, model, index_path, inf_save_path, k=8):
    ###Now, embed the query_dataset
    query_embeddings,_ = get_all_embeddings(query_dataset, model)

    ##Convert to numpy
    query_embeddings = query_embeddings.cpu().numpy()

    index=faiss.read_index(index_path)

    ###ref dataset path dict
    ref_dataset_path_dict=ref_dataset.subsetted_path_dict
    ####Search the embeddings
    _, indices = index.search(query_embeddings, k=k)

    # ####Now, for each index in indices, get the word for the ref path 
    all_nns = []
    for i, idx in enumerate(tqdm(indices)):
        ###use path dict to get the path
        nn_paths = [ref_dataset_path_dict[j][0] for j in idx]
        nn_words = [os.path.basename(path).split("-word-")[1] for path in nn_paths]
        nn_words = [word.split(".")[0] for word in nn_words]
        all_nns.append("|".join(nn_words))

    with open(inf_save_path, 'w') as f:
        f.write("\n".join(all_nns))

def prepare_hn_query_paths(query_paths,train_dataset,paired_hn=True,font_paths=[],max_word_n=40,image_size=224):
    if paired_hn:
        query_paths = [x[0] for x in train_dataset.data if "PAIRED" in os.path.basename(x[0])]
    else:
        query_paths = [x[0] for x in train_dataset.data]
        ###Keep only those paths that contain any of the fonts in font_paths
        query_paths = [x for x in query_paths if any([font in x for font in font_paths])]


    print("Number of query paths: ", len(query_paths))
    ###Get the list of directory names from the query_paths

    if paired_hn:
        ##Get paired paths
        query_paths = [x[0] for x in train_dataset.data if "PAIRED" in os.path.basename(x[0])]
        unpaired_paths=[x[0] for x in train_dataset.data if "PAIRED" not in os.path.basename(x[0]) and args.default_font_name in os.path.basename(x[0])]
        ####Get only one unpaired path per word - dedup
        unpaired_paths_dedup = []
        unpaired_path_words = [os.path.basename(x).split("-word-")[1].split(".")[0] for x in unpaired_paths]
        unpaired_path_words_unique = list(set(unpaired_path_words))
        ###We only want one path per word from the unpaired_paths
        for word in unpaired_path_words_unique:
            unpaired_paths_dedup.append(unpaired_paths[unpaired_path_words.index(word)])

        unpaired_paths = unpaired_paths_dedup

        print(f"Num unpaired paths: {len(unpaired_paths)}")

    

    # ###Now, we want to take at most 10 paired paths per word
    ##First, let's make a dict of word to paths
    print("preparing word paths dict")
    word_to_paths = defaultdict(list)
    for path in tqdm(query_paths):
        word_to_paths[os.path.basename(path).split("-word-")[1].split(".")[0]].append(path)
    
    ###Now, we want to take at most max_word_n paths per word using the word_to_paths dict
    max_word_n_paths = []
    for word in word_to_paths.keys():
        if len(word_to_paths[word]) <= max_word_n:
            max_word_n_paths.extend(word_to_paths[word])
        else:
            ##Shuffle the paths
            np.random.shuffle(word_to_paths[word])
            max_word_n_paths.extend(word_to_paths[word][:max_word_n])


    paired_paths = max_word_n_paths

    print(f"Num selected paths ({max_word_n} at max): {len(paired_paths)}")


    if paired_hn:            
        query_paths = list(set(paired_paths + unpaired_paths))
    else:
        query_paths = list(set(paired_paths))

    ###save query paths to file
    with open(os.path.join(args.run_name, f"query_paths.txt"), "w") as f:
        for path in query_paths:
            f.write(f"{path}\n")

    query_dataset = create_hn_query_dataset(args.root_dir_path, imsize=image_size,hn_query_list=query_paths)


    print(f"Num hard neg paths: {len(query_paths)}")    
    return query_paths, query_dataset

def save_ref_index(ref_dataset, model, save_path,prefix=""):

    os.makedirs("indices", exist_ok=True)
    knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP, reset_before=False, reset_after=False)
    infm = InferenceModel(model, knn_func=knn_func)
    infm.train_knn(ref_dataset)
    infm.save_knn_func(os.path.join(save_path, "ref.index"))

    ref_data_file_names = [os.path.basename(x[0]).split("-word-")[1].split(".")[0]  for x in ref_dataset.data]
    with open(os.path.join(save_path, f"{prefix}ref.txt"), "w") as f:
        f.write("\n".join(ref_data_file_names))

def save_model(model_folder, enc, epoch, datapara):

    if not os.path.exists(model_folder): os.makedirs(model_folder)

    if datapara:
        torch.save(enc.module.state_dict(), os.path.join(model_folder, f"enc_{epoch}.pth"))
    else:
        torch.save(enc.state_dict(), os.path.join(model_folder, f"enc_{epoch}.pth"))

def get_all_embeddings(dataset, model, batch_size=128):

    tester = testers.BaseTester(batch_size=batch_size)
    
    return tester.get_all_embeddings(dataset, model)

def tester_knn(test_set, ref_set, model, accuracy_calculator, split, log=False):

    model.eval()

    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    test_labels = test_labels.squeeze(1)
    ref_embeddings, ref_labels = get_all_embeddings(ref_set, model)
    ref_labels = ref_labels.squeeze(1)

    print("Computing accuracy...")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
        ref_embeddings,
        test_labels,
        ref_labels,
        embeddings_come_from_same_source=False)
    

    prec_1 = accuracies["precision_at_1"]

    ##Log the accuracy
    if log:
        wandb.log({f"{split}/accuracy": prec_1})
    print(f"Accuracy on {split} set (Precision@1) = {prec_1}")
    return prec_1

def trainer_knn_with_eval(model, loss_func, device, train_loader, optimizer, epoch, epochviz=None, diff_sizes=False,scheduler=None,int_eval_steps=None,zs_accuracy=0,wandb_log=False):

    model.train()

    for batch_idx, (data, labels) in enumerate(train_loader):

        labels = labels.to(device)
        data = [datum.to(device) for datum in data] if diff_sizes else data.to(device)

        optimizer.zero_grad()

        if diff_sizes:
            out_emb = []
            for datum in data:
                emb = model(datum.unsqueeze(0)).squeeze(0)
                out_emb.append(emb)
            embeddings = torch.stack(out_emb, dim=0)
        else:
            embeddings = model(data)

        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()

        if wandb_log:
            wandb.log({"train/loss": loss.item()})

        if int_eval_steps!=None:
            if batch_idx % int_eval_steps == 0:
                acc = tester_knn(val_dataset, render_dataset, model, accuracy_calculator, "val",log=wandb_log)
                print("Intermediate accuracy: ",acc)
                if wandb_log:
                    wandb.log({"val/acc": acc})
                if acc>zs_accuracy:
                    save_model(args.run_name, model, "best_cer", datapara)
                    zs_accuracy=acc

        if batch_idx % 100 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(str(epoch).zfill(3), str(batch_idx).zfill(4), loss))
            if not epochviz is None:
                for i in range(10):
                    image = T.ToPILImage()(INV_NORMALIZE(data[i].cpu()))
                    image.save(os.path.join(epochviz, f"train_sample_{epoch}_{i}.png"))

        del embeddings
        del loss
        del labels
        if scheduler!=None:
            scheduler.step()
            if wandb_log:
                wandb.log({"train/lr": scheduler.get_lr()[0]})
    return zs_accuracy
               
if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir_path", type=str, required=True,
        help="Root image directory path, with character class subfolders")
    parser.add_argument("--font_dir_path", type=str, default=None)
    parser.add_argument("--word_dict", type=str, default=None)
    parser.add_argument("--ascender", action='store_true', default=True)
    parser.add_argument("--train_ann_path", type=str, required=True,
        help="Path to COCO-style annotation file that localizer was trained on")
    parser.add_argument("--val_ann_path", type=str, required=True,
        help="Path to COCO-style annotation file that localizer was validated on")
    parser.add_argument("--test_ann_path", type=str, required=True,
        help="Path to COCO-style annotation file that localizer was tested on")
    parser.add_argument("--train_mode", type=str, required=True, choices=["character", "word"])
    parser.add_argument("--run_name", type=str, required=True,
        help="Name of run for W&B logging purposes. Also the name of the folder where checkpoints will be saved")
    parser.add_argument('--batch_size', type=int, default=128,
        help="Batch size")
    parser.add_argument('--lr', type=float, default=2e-6,
        help="LR for AdamW")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
        help="Weight decay for AdamW")
    parser.add_argument('--num_epochs', type=int, default=5,
        help="Number of epochs")
    parser.add_argument('--temp', type=float, default=0.1,
        help="Temperature for Supcon loss")
    parser.add_argument('--start_epoch', type=int, default=1,
        help="Starting epoch")
    parser.add_argument('--m', type=int, default=4,
        help="m for m in m-class sampling")
    parser.add_argument('--imsize', type=int, default=224,
        help="Size of image for encoder")
    parser.add_argument("--hns_txt_path", type=str, default=None,
        help="Path to text file of mined hard negatives")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Load checkpoint before training")
    parser.add_argument('--finetune', action='store_true', default=False,
        help="Train just on target character crops")
    parser.add_argument('--pretrain', action='store_true', default=False,
        help="Train just on render character crops")
    parser.add_argument('--high_blur', action='store_true', default=False,
        help="Increase intensity of the blurring data augmentation for renders")
    parser.add_argument('--diff_sizes', action='store_true', default=False,
        help="DEPRECATED: allow different sizes for training crops")
    parser.add_argument('--epoch_viz_dir', type=str, default=None,
        help="Visualize and save some training samples by batch to this directory")
    parser.add_argument('--infer_hardneg_k', type=int, default=8,
        help="Infer k-NN hard negatives for each training sample, and save to a text file")
    parser.add_argument('--test_at_end', action='store_true', default=False,
        help="Inference on test set at end of training with best val checkpoint")
    parser.add_argument("--auto_model_hf", type=str, default=None,
        help="Use model from HF by specifying model name")
    parser.add_argument("--auto_model_timm", type=str, default=None,
        help="Use model from timm by specifying model name")
    parser.add_argument("--num_passes", type=int, default=1,
        help="Defines epoch as number of passes of N_chars * M. Only for train_mode=character")
    parser.add_argument('--no_aug', action='store_true', default=False,
        help="Turn of data augmentation")
    parser.add_argument('--lr_schedule', action='store_true', default=False)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--dec_lr_factor', default=0.9)
    parser.add_argument('--adamw_beta1', type=float, default=0.9)
    parser.add_argument('--adamw_beta2', type=float, default=0.999)
    parser.add_argument('--char_only_sampler', action='store_true', default=False)
    parser.add_argument('--aug_paired', action='store_true', default=False)
    parser.add_argument('--expansion_factor', type=int, default = 1,help="Expansion factor is in beta, use at your own risk")
    parser.add_argument('--int_eval_steps', type=int, default=None)
    parser.add_argument('--wandb_log',action='store_true', default=False)
    parser.add_argument('--default_font_name',type=str, default="Noto")
    args = parser.parse_args()


    ###Prepare synthetic and image folder
    ###WE generate synthetic images and put paired images in the corresponding folder for the word or character

    render_all_synth_in_parallel(args.root_dir_path, args.font_dir_path, args.word_dict, args.ascender)

    ###send the paired images for each word in the right subfolder in the root directory. Then, create an annnotation file of the format: 
    # {"images": [{"file_name": "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress/silver_lines/words/100_111_117_98_108_105_110_103/PAIRED-sn86076201_00279554048_1916040101_0685_6_14_2-word-100_111_117_98_108_105_110_103.jpg"}, {"file_name": "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress/silver_lines/words/100_111_117_98_108_105_110_103/PAIRED-sn95066012_THIRTEENTH YEAR_3721_1_19_12_1-word-100_111_117_98_108_105_110_103.jpg"}, {"file_name": "/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/e2e2e/end-to-end-pipeline/pipeline_egress/silver_lines/words/100_111_117_98_108_105_110_103/PAIRED-sn83030272_0020653347A_1876040101_0897_9_25_6-word-100_111_117_98_108_105_110_103.jpg"},

    ###In the end, we have the root folder specified by the path by the user (or a default path) with subfolders for each word. Each subfolder contains the paired images for that word.
    ###The annotation files are 3 dicts for train, test and val. So those arguments can be knocked off. And a copy can be saved for examining. 

    # setup
    if args.wandb_log:
        wandb.init(project="effocr_recog_v2", name=args.run_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.run_name, exist_ok=True)
    with open(os.path.join(args.run_name, "args_log.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # load encoder

    if args.auto_model_hf is None and args.auto_model_timm is None:
        raise NotImplementedError
    elif not args.auto_model_timm is None:
        encoder = AutoEncoderFactory("timm", args.auto_model_timm)
    elif not args.auto_model_hf is None:
        encoder = AutoEncoderFactory("hf", args.auto_model_hf)

    # init encoder

    if args.checkpoint is None:
        if not args.auto_model_timm is None:
            enc = encoder(args.auto_model_timm)
        elif not args.auto_model_hf is None:
            enc = encoder(args.auto_model_hf)
        else:
            enc = encoder()
    else:
        enc = encoder.load(args.checkpoint)

    # data parallelism

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        datapara = True
        enc = nn.DataParallel(enc)
    else:
        datapara = False
    
    # create dataset


    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, num_batches  = create_dataset(
        args.root_dir_path, 
        args.train_ann_path,
        args.val_ann_path, 
        args.test_ann_path, 
        args.batch_size,
        hardmined_txt=args.hns_txt_path, 
        train_mode=args.train_mode,
        m=args.m, 
        finetune=args.finetune,
        pretrain=args.pretrain,
        high_blur=args.high_blur,
        diff_sizes=args.diff_sizes,
        imsize=args.imsize,
        num_passes=args.num_passes,
        no_aug=args.no_aug,
        k=args.k,
        aug_paired=args.aug_paired,
        expansion_factor=args.expansion_factor,
    )

    render_dataset = create_render_dataset(
        args.root_dir_path,
        train_mode=args.train_mode,
        font_name=args.default_font_name,
        imsize=args.imsize,
    )

    
    
    # optimizer and loss

    optimizer = AdamW(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.adamw_beta1, args.adamw_beta2))


    loss_func = losses.SupConLoss(temperature = args.temp) 

    # get zero-shot accuracy
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)

    if args.checkpoint is not None:
        print("Zero-shot accuracy:")
        best_acc=  tester_knn(val_dataset, render_dataset, enc, accuracy_calculator, "zs",log=args.wandb_log)
        ##Log 
        if args.wandb_log:
            wandb.log({"val/acc": best_acc})


    if num_batches is None:
        t_0 = 1000 
    else:
        t_0 = num_batches
    


    # set  schedule

    if args.lr_schedule:
        scheduler=CosineAnnealingDecWarmRestarts(optimizer, T_0=t_0, T_mult=2,l_dec=0.9) 
    else:
        scheduler=None
    # warm start training

    print("Training...")
    if not args.epoch_viz_dir is None: os.makedirs(args.epoch_viz_dir, exist_ok=True)
    for epoch in range(args.start_epoch, args.num_epochs+args.start_epoch):
            acc=trainer_knn_with_eval(enc, loss_func, device, train_loader, optimizer, epoch, args.epoch_viz_dir, args.diff_sizes,scheduler,int_eval_steps=args.int_eval_steps,zs_accuracy=best_acc if best_acc != None else 0,wandb_log=args.wandb_log)
            acc = tester_knn(val_dataset, render_dataset, enc, accuracy_calculator, "val",log=args.wandb_log)
            ##Log
            if args.wandb_log:
                wandb.log({"val/acc": acc})

            if acc >= best_acc:
                best_acc = acc
                print("Saving model and index...")
                save_model(args.run_name, enc, "best", datapara)
                print("Model and index saved.")

                if scheduler!=None:
                    scheduler.step()
                    ###Log on wandb
                    if args.wandb_log:
                        wandb.log({"train/lr": scheduler.get_last_lr()[0]})

    del enc
    best_enc = encoder.load(os.path.join(args.run_name, "enc_best.pth"))
    save_ref_index(render_dataset, best_enc, args.run_name)


    if args.test_at_end:
        print("Testing on test set...")
        tester_knn(test_dataset, render_dataset, best_enc, accuracy_calculator, "test")
        print("Test set testing complete.")


    # optionally infer hard negatives (turned on by default, highly recommend to facilitate hard negative training)

    if not args.infer_hardneg_k is None :
        query_paths = [x[0] for x in train_dataset.data if os.path.basename(x[0])]
        print("Number of query paths: ", len(query_paths))
        query_paths,query_dataset=prepare_hn_query_paths(query_paths,train_dataset,paired_hn=True,image_size=args.imsize)
        print(f"Num hard neg paths: {len(query_paths)}")    
        transform = create_paired_transform(args.imsize)
        infer_hardneg_dataset(query_dataset, train_dataset if args.finetune else render_dataset, best_enc, 
            os.path.join(args.run_name, "ref.index"), os.path.join(args.run_name, "hns.txt"), 
            k=args.infer_hardneg_k)