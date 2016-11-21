weak-1 train --txt "../data/train/txt/record-19.txt" --annotations "../data/train/con/*" --model ../models/weak1-train.model --format i2b2

weak-1 predict --txt "../data/train/txt/record-19.txt" --model ../models/weak1-train.model --out ../data/predictions/weak1-demo --format i2b2

weak-1 evaluate --predictions ../data/predictions/weak1-demo/ --gold ../data/train/con/ --format i2b2
