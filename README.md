# AR/VR Predictive Caching

Run split_train_validation_data.py to create Train, Test, Validation datasets from the original dataset.

Run these steps 3 times to create Train, Test, and validation inputs:

1. Check the setting.py for "house1" or "house2" settings and "train"/"validation"/"test" data generation
2. Run dataset_CASAS_reader.py which creates lists of daily tasks based on the selected dataset in setting.py.
3. Run execute.py for data generation (optional).
4. Run create_requests_dataset.py which creates synthetic dataset.
5. Run create_LSTM_input.py that converts the input to one-hot encoding format.

After creating the inputs:
Run LSTM.py to train the LSTM network.
Run caching.py to calculate and print caching costs and experience scores for each method. 

## Datasets

[Dataset 1](http://sites.google.com/site/tim0306/): 
T. L. van  Kasteren, G. Englebienne, and B. J.  Kr ̈ose, “Human activityrecognition  from  wireless  sensor  network  data:  Benchmark  and  soft-ware,”  in Activity  Recognition  in  Pervasive  Intelligent  Environments.Springer, 2011, pp. 165–186.

The dataset describes the activities of a 26-year-old man in a smart home with 14 state-change sensors installed at doors, cupboards, the refrigerator, and the toilet flush.

[Dataset 2](http://casas.wsu.edu/datasets/): 
D. J. Cook, A. S. Crandall, B. L. Thomas, and N. C. Krishnan, “CASAS:A smart home in a box,”Computer, vol. 46, no. 7, pp. 62–69, 2013.

We used dataset #14 (Cairo). This dataset collected by CASAS research group describes the activities of 2 residents in an apartment for 57 days.

## Citation

Link: [Here](https://ieeexplore.ieee.org/abstract/document/9049692)

@inproceedings{zehtabian2020predictive,

  title={Predictive Caching for AR/VR Experiences in a Household Scenario},  
  author={Zehtabian, Sharare and Razghandi, Mina and B{\"o}l{\"o}ni, Ladislau and Turgut, Damla}  
  booktitle={2020 International Conference on Computing, Networking and Communications (ICNC)},  
  pages={591--595},  
  year={2020}
}
