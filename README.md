# AR/VR Predictive Caching

Run split_train_validation_data.py to create Train, Test, Validation datasets from the original dataset.

Run these steps 3 times to create Train, Test, and validation inputs:

1. Check the setting.py for "house1" or "house2" settings and "train"/"validation"/"test" data generation
2. Run dataset_CASAS_reader.py which creates lists of daily tasks based on the selected dataset in setting.py.
3. Run execute.py for data generation (optional).
4. Run create_requests_dataset.py which created synthetic dataset.
5. Run create_LSTM_input.py that converts the input to one-hot encoding.
6. Run LSTM.py to train the LSTM network.
7. Run caching.py to print caching costs and experience scores for each method. 
