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

## Citation

@inproceedings{zehtabian2020predictive,
  title={Predictive Caching for AR/VR Experiences in a Household Scenario},  
  author={Zehtabian, Sharare and Razghandi, Mina and B{\"o}l{\"o}ni, Ladislau and Turgut, Damla}  
  booktitle={2020 International Conference on Computing, Networking and Communications (ICNC)},  
  pages={591--595},  
  year={2020}
}
