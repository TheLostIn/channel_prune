# channel_prune
Base to channel pruned to ResNet18 model

PyTorch implementation

This demonstrates pruning a ResNet18 based multi-classification.

This was able to reduce the CPU runtime by x2 and the model size by x2 at least.

For more details you can read the paper:Pruning Convolutional Neural Networks for Resource Efficient Inference,
https://arxiv.org/abs/1611.06440.

At each pruning step 512 filters are removed from the network, the number was set by yourself.

Usage

Training: python finetune.py --train --train_path /path/train --test_path /path/test

Pruning:  python finetune.py --prune --train_path /path/train --test_path /path/test

Testing:  python finetune.py --test --train_path /path/train --test_path /path/test

note

Change the pruning to be done in one pass. Currently each of the 512 filters are pruned sequentually. for layer_index, filter_index in prune_targets: model = prune_vgg16_conv_layer(model, layer_index, filter_index)

This is inefficient since allocating new layers, especially fully connected layers with lots of parameters, is slow.

In principle this can be done in a single pass.

torch version:0.1.12_2 (recommendation)

this project modified from https://github.com/eeric/channel_prune.

For details, see csdn: http://blog.csdn.net/yyqq7226741/article/details/78301231

## modify detail

1. change `self.model.features._modules.items()[layer][1][kt].conv1(x)` into `self.model.features._modules.get(str(layer))[kt].conv1(x)`

2. change `    `  into `	`，the indent method.

3. added `test` mode.
