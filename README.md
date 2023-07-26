1, This repository contains the Material of the article Supplementary material for the article ["Predictive Business Process Monitoring with
LSTM Neural Networks"](https://arxiv.org/abs/1612.02130) by Niek Tax, Ilya Verenich, [Marcello La Rosa](http://www.marcellolarosa.com/) and [Marlon Dumas](http://kodu.ut.ee/~dumas/).

The files alculate_accuracy_on_next_event.py, evaluate_next_activity_and_time.py, evaluate_suffix_and_remaining_time.py and train.py do the following:

- Prediction of the next type of activity to be executed in a running process instance
- Prediction of the timestamp of the next type of activity to be executed
- Prediction of the continuation of a running instance, i.e. its suffix
- Prediction of the remaining cycle time of an instance

The results of this are stored in output_files

For more details regarding the PREDICTIONS, please refer to the [project page](https://verenich.github.io/ProcessSequencePrediction)

2, On the Predictions the ADJUSTMENT FOR REGRESSION TOWARDS THE MEAN has been done in the files helpdesk.py and env_permit.py.

Both scripts produce four graphs which are shown in the seminar paper, in main directory (Code). On the helpdesk dataset there needs to be four lines of code commented out, in order to produce the graph for the filtered dataset, which is also written in the script
The graphs can also be found in the folder MyResults without running the scripts.
