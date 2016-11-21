weak-1 train --txt "../data/train/txt/*.txt" --annotations "../data/train/con/*" --model ../models/weak1-train.model --format i2b2

weak-1 predict --txt "../data/test/txt/*.txt" --model ../models/weak1-train.model --out ../data/predictions/weak1-test --format i2b2

weak-1 predict --txt "../data/train/txt/*.txt" --model ../models/weak1-train.model --out ../data/predictions/weak1-train --format i2b2

weak-1 evaluate --predictions ../data/predictions/weak1-test/ --gold ../data/test/con/ --format i2b2

weak-1 evaluate --predictions ../data/predictions/weak1-train/ --gold ../data/train/con/ --format i2b2
