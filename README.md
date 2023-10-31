# CGNN
### A graph convolutional neuron network designed for predicting crystalline material properties.

## Quick Run
* To train a model:

  `python main.py --root_dir=./tests_train` 

* To predict:

  `python main.py --predict --root_dir=./test_predict`

* To change the settings of training:

  ```
  python main.py --root_dir=./test_train --epochs=100 --train_ratio=0.6 --val_ratio=0.2 --test_ratio=0.2`

  python main.py --root_dir=./test_train --dropout_rate=0.5 --loss_func=CrossEntropyLoss --optimizer=RMSprop
  ```

## Input Format
**In the 'training' mode**, the directory `root_dir` should include .cif files of all the crystals, and a targets.csv file in which the first column contains the names of .cif files, and the second column contains the target values.

**In the 'predict' mode**, only .cif files should be provided. **DO NOT include a targets.csv file**.

Examples can be found in tests_train and test_predict folders.

## References
https://www.cell.com/patterns/fulltext/S2666-3899(22)00076-9#%20

[https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301
