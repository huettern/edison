# kws training

The files in here are responsible for training different keyword spotting networks on different data and framework.

Before running, make sure the parent directory is in the python paht:
```bash
cd ../
export PYTHONPATH=`pwd`:$PYTHONPATH
cd train/
```

## kws_legacy

`kws_legacy.py` contains initial tries and is no longer used.

## kws_keras

`kws_keras.py` uses the keras framework to train a model for multi-keyword spotting.

### Models

Inspired by https://github.com/tensorflow/tensorflow/blob/5d595477cc94a78c364e81810beb3753ae756607/tensorflow/examples/speech_commands/models.py#L673

| Name                  | Total params | MACC       | ROM                    | RAM      | Exec time          |
|:----------------------|:-------------|:-----------|:-----------------------|:---------|:-------------------|
| conv_model            | 202,884      | 22,567,292 | 792KB                  | 39KiB    | 1623ms -> too slow |
| low_latency_conv      | 3,015,662    | ?          | 11.5MB -> does not fit | 39KiB    | ?                  |
| single_fc             | 1,616        | 1,672      | 6.46 KB                | 0.02 KiB | 0.206ms            |
| tiny_conv             | 4,236        | 76,228     | 16.97 KB               | 3.6 KB   | 8.392ms            |
| tiny_embedding_conv   | 5,812        | 78,836     | 23.25 KB               | 1.47 KB  | 8.789ms            |
| medium_embedding_conv | 5,664        | 218,840    | 22.66 KB               | 3.42 KB  | 22.202ms           |
