import numpy as np
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description='split a file')
    parser.add_argument(
        '--file', default="/home/zhangdongyu/object-detection/data/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt", type=str)
    parser.add_argument(
        '--save', default="/home/zhangdongyu/object-detection/data/VOCdevkit2007/VOC2007/ImageSets/Main/test/", type=str)
    parser.add_argument(
        '--size', default=50, type=int)
    parser.add_argument(
        '--work', default=0, type=int)
    
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    if args.work==1:
        text = []
        with open(args.file, 'r') as f:
            for x in f:
                text.append(x)
        random.shuffle(text)
        for i in range((len(text)+args.size-1)//args.size   ):
            with open(args.save+"test{:d}.txt".format(i), 'w') as f:
                for x in text[i*args.size:min((i+1)*args.size, len(text)-1)]:
                    f.write(x)




